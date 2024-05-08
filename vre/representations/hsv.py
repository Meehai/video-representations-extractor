"""HSV module"""
import numpy as np
from overrides import overrides
from skimage.color import rgb2hsv
from ..representation import Representation, RepresentationOutput
from ..utils import image_resize_batch

class HSV(Representation):
    """HSV representation"""
    @overrides
    def make(self, frames: np.ndarray) -> RepresentationOutput:
        return rgb2hsv(frames).astype(np.float32)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        return (repr_data * 255).astype(np.uint8)

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return image_resize_batch(repr_data, height=new_size[0], width=new_size[1])
