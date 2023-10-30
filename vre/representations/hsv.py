"""HSV module"""
import numpy as np
from overrides import overrides
from skimage.color import rgb2hsv
from ..representation import Representation


class HSV(Representation):
    """HSV representation"""
    @overrides
    def make(self, t: slice) -> np.ndarray:
        return np.float32(rgb2hsv(np.array(self.video[t])))

    @overrides
    def make_image(self, x: np.ndarray) -> np.ndarray:
        return np.uint8(x * 255)
