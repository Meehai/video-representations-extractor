import numpy as np
from overrides import overrides
from ..representation import Representation, RepresentationOutput


class RGB(Representation):
    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        return np.float32(self.video[t]) / 255

    @overrides
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        return np.uint8(x * 255)
