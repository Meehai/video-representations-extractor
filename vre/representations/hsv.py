"""HSV module"""
import numpy as np
from overrides import overrides
from skimage.color import rgb2hsv
from ..representation import Representation, RepresentationOutput


class HSV(Representation):
    """HSV representation"""
    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        return np.float32(rgb2hsv(np.array(self.video[t])))

    @overrides
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        return np.uint8(x * 255)
