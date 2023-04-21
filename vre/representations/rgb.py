import numpy as np
from overrides import overrides
from .representation import Representation, RepresentationOutput


class RGB(Representation):
    @overrides
    def make(self, t: int) -> RepresentationOutput:
        return np.float32(self.video[t]) / 255

    @overrides
    def make_image(self, x: RepresentationOutput) -> np.ndarray:
        return np.uint8(x["data"] * 255)

    @overrides
    def setup(self):
        pass
