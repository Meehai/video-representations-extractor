import numpy as np
import cv2
from overrides import overrides
from matplotlib.cm import gray
from ...representation import Representation, RepresentationOutput


class Canny(Representation):
    def __init__(self, threshold1: float, threshold2: float, apertureSize: int, L2gradient: bool, **kwargs):
        super().__init__(**kwargs)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.apertureSize = apertureSize
        self.L2gradient = L2gradient

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        raise NotImplementedError
        frame = self.video[t]
        res = frame * 0
        res = cv2.Canny(frame, self.threshold1, self.threshold2, res, self.apertureSize, self.L2gradient)
        res = np.float32(res) / 255
        return res

    @overrides
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        return np.uint8(255 * gray(x["data"])[..., 0:3])
