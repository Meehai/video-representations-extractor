import numpy as np
from overrides import overrides
from ..representation import Representation


class RGB(Representation):
    @overrides
    def make(self, t: int) -> np.ndarray:
        return np.float32(self.video[t]) / 255

    @overrides
    def make_image(self, x: np.ndarray) -> np.ndarray:
        return np.uint8(x["data"] * 255)
