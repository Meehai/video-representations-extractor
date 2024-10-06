"""Canny edge detector representation."""
import numpy as np
from overrides import overrides
from matplotlib.cm import gray # pylint: disable=no-name-in-module
from ...representation import Representation, RepresentationOutput
from ...utils import image_resize_batch
from ...utils.cv2_utils import cv2_Canny

class Canny(Representation):
    """Canny edge detector representation."""
    def __init__(self, threshold1: float, threshold2: float, aperture_size: int, l2_gradient: bool, **kwargs):
        super().__init__(**kwargs)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, RepresentationOutput] | None = None) -> RepresentationOutput:
        return RepresentationOutput(output=np.array([self._make_one(frame) for frame in frames]))

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        return (255 * gray(repr_data.output)[..., 0:3]).astype(np.uint8)

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return RepresentationOutput(output=image_resize_batch(repr_data.output, *new_size))

    def _make_one(self, x: np.ndarray) -> np.ndarray:
        res = cv2_Canny(x, threshold1=self.threshold1, threshold2=self.threshold2,
                        apertureSize=self.aperture_size, L2gradient=self.l2_gradient)
        res = np.float32(res) / 255
        return res

    @overrides
    def vre_setup(self):
        pass
