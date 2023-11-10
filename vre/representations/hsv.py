"""HSV module"""
import numpy as np
from overrides import overrides
from skimage.color import rgb2hsv
from ..representation import Representation, RepresentationOutput


class HSV(Representation):
    """HSV representation"""
    @overrides
    def vre_setup(self, **kwargs):
        pass

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        return rgb2hsv(np.array(self.video[t])).astype(np.float32)

    @overrides
    def make_images(self, t: slice, x: np.ndarray, extra: dict | None) -> np.ndarray:
        return (x * 255).astype(np.uint8)
