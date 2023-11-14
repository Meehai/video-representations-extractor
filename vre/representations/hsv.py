"""HSV module"""
import numpy as np
from overrides import overrides
from skimage.color import rgb2hsv
from ..representation import Representation, RepresentationOutput


class HSV(Representation):
    """HSV representation"""
    @overrides
    def make(self, frames: np.ndarray) -> RepresentationOutput:
        return rgb2hsv(frames).astype(np.float32)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        return (repr_data * 255).astype(np.uint8)
