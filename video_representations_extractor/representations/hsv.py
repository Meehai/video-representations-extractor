import numpy as np
from overrides import overrides
from skimage.color import rgb2hsv
from .representation import Representation, RepresentationOutput

class HSV(Representation):
    @overrides
    def make(self, t: int) -> RepresentationOutput:
        return np.float32(rgb2hsv(self.video[t]))
         
    @overrides
    def makeImage(self, x: RepresentationOutput) -> np.ndarray:
        return np.uint8(x["data"] * 255)

    @overrides
    def setup(self):
        pass
