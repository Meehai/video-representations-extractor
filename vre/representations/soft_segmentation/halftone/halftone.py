"""
Halftone representation
The bulk of this is taken from this Stack Overflow answer by fraxel: http://stackoverflow.com/a/10575940/250962
Copy pasted from https://github.com/philgyford/python-halftone/blob/main/halftone.py in order to change constructor
"""

import numpy as np

from PIL import Image, ImageDraw, ImageStat
from overrides import overrides

from ....representation import Representation, RepresentationOutput
from ....utils import image_resize, image_resize_batch


class Halftone(Representation):
    """
    Halftone representation
    Parameters:
    - sample: Sample box size from original image, in pixels.
    - scale: Max output dot diameter is sample * scale (which is also the number of possible dot sizes)
    - percentage: How much of the gray component to remove from the CMY channels and put in the K channel.
    - angles: A list of 4 angles that each screen channel should be rotated by.
    - antialias: boolean.
    """

    def __init__(self, sample: float, scale: float, percentage: float, angles: list[int],
                 antialias: bool, resolution: tuple[int, int], **kwargs):
        super().__init__(**kwargs)
        assert len(resolution) == 2, resolution
        self.sample = sample
        self.scale = scale
        self.percentage = percentage
        self.angles = angles
        self.antialias = antialias
        self.resolution = resolution
        self._check_arguments()

    @overrides
    def make(self, frames: np.ndarray) -> RepresentationOutput:
        return np.array([self._make_one_image(frame) for frame in frames])

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        return (repr_data * 255).astype(np.uint8)

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return image_resize_batch(repr_data, height=new_size[0], width=new_size[1])

    def _gcr(self, im, percentage):
        """
        Basic "Gray Component Replacement" function. Returns a CMYK image with
        percentage gray component removed from the CMY channels and put in the
        K channel, ie. for percentage=100, (41, 100, 255, 0) >> (0, 59, 214, 41)
        """
        cmyk_im = im.convert("CMYK")
        if not percentage:
            return cmyk_im
        cmyk_im = cmyk_im.split()
        cmyk = []
        for i in range(4):
            cmyk.append(cmyk_im[i].load())
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                gray = int(min(cmyk[0][x, y], cmyk[1][x, y], cmyk[2][x, y]) * percentage / 100)
                for i in range(3):
                    cmyk[i][x, y] = cmyk[i][x, y] - gray
                cmyk[3][x, y] = gray
        return Image.merge("CMYK", cmyk_im)

    def _halftone(self, im, cmyk):
        """
        Returns list of half-tone images for cmyk image. sample (pixels),
        determines the sample box size from the original image. The maximum
        output dot diameter is given by sample * scale (which is also the number
        of possible dot sizes). So sample=1 will presevere the original image
        resolution, but scale must be >1 to allow variation in dot size.
        """

        # If we're antialiasing, we'll multiply the size of the image by this
        # scale while drawing, and then scale it back down again afterwards.
        # Because drawing isn't aliased, so drawing big and scaling back down
        # is the only way to get antialiasing from PIL/Pillow.
        antialias_scale = 4

        scale = self.scale
        if self.antialias is True:
            scale *= antialias_scale

        cmyk = cmyk.split()
        dots = []

        for channel, angle in zip(cmyk, self.angles):
            channel = channel.rotate(angle, expand=1)
            size = channel.size[0] * scale, channel.size[1] * scale
            half_tone = Image.new("L", size)
            draw = ImageDraw.Draw(half_tone)

            # Cycle through one sample point at a time, drawing a circle for
            # each one:
            for x in range(0, channel.size[0], self.sample):
                for y in range(0, channel.size[1], self.sample):

                    # Area we sample to get the level:
                    box = channel.crop((x, y, x + self.sample, y + self.sample))

                    # The average level for that box (0-255):
                    mean = ImageStat.Stat(box).mean[0]

                    # The diameter of the circle to draw based on the mean (0-1):
                    diameter = (mean / 255) ** 0.5

                    # Size of the box we'll draw the circle in:
                    box_size = self.sample * scale

                    # Diameter of circle we'll draw:
                    # If sample=10 and scale=1 then this is (0-10)
                    draw_diameter = diameter * box_size

                    # Position of top-left of box we'll draw the circle in:
                    # x_pos, y_pos = (x * scale), (y * scale)
                    box_x, box_y = (x * scale), (y * scale)

                    # Positioned of top-left and bottom-right of circle:
                    # A maximum-sized circle will have its edges at the edges
                    # of the draw box.
                    x1 = box_x + ((box_size - draw_diameter) / 2)
                    y1 = box_y + ((box_size - draw_diameter) / 2)
                    x2 = x1 + draw_diameter
                    y2 = y1 + draw_diameter

                    draw.ellipse([(x1, y1), (x2, y2)], fill=255)

            half_tone = half_tone.rotate(-angle, expand=1)
            width_half, height_half = half_tone.size

            # Top-left and bottom-right of the image to crop to:
            xx1 = (width_half - im.size[0] * scale) / 2
            yy1 = (height_half - im.size[1] * scale) / 2
            xx2 = xx1 + im.size[0] * scale
            yy2 = yy1 + im.size[1] * scale

            half_tone = half_tone.crop((xx1, yy1, xx2, yy2))

            if self.antialias is True:
                # Scale it back down to antialias the image.
                w = int((xx2 - xx1) / antialias_scale)
                h = int((yy2 - yy1) / antialias_scale)
                half_tone = image_resize(half_tone, h, w)

            dots.append(half_tone)
        return dots

    def _make_one_image(self, frame: np.ndarray) -> np.ndarray:
        frame = image_resize(frame, height=self.resolution[0], width=self.resolution[1])
        im = Image.fromarray(frame, "RGB")
        cmyk = self._gcr(im, self.percentage)
        channel_images = self._halftone(im, cmyk)
        new = Image.merge("CMYK", channel_images)
        new = np.array(new)[..., 0:3]
        new = np.float32(new) / 255
        return new

    def _check_arguments(self):
        assert len(self.angles) == 4, f"The angles argument must be a list of 4 integers, but it has {len(self.angles)}"
        for a in self.angles:
            assert isinstance(a, int), f"All four elements of the angles list must be integers, got: {self.angles}"
        assert isinstance(self.antialias, bool), f"The antialias argument must be a boolean, not '{self.antialias}'."
        assert isinstance(self.percentage, (float, int)), \
            f"The percentage argument must be an integer or float, not '{self.percentage}'."
        assert isinstance(self.sample, int), f"The sample argument must be an integer, not '{self.sample}'."
        assert isinstance(self.scale, int), f"The scale argument must be an integer, not '{self.scale}'."
