import numpy as np
from overrides import overrides
from ..representation import Representation, RepresentationOutput


class RGB(Representation):
    @overrides
    def vre_setup(self, **kwargs):
        pass

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        return np.array(self.video[t]).astype(np.float32) / 255

    @overrides
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        return (x * 255).astype(np.uint8)
