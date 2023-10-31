import numpy as np
import cv2
from overrides import overrides
from matplotlib.cm import gray
from ...representation import Representation, RepresentationOutput


class Canny(Representation):
    def __init__(self, threshold1: float, threshold2: float, aperture_size: int, l2_gradient: bool, **kwargs):
        super().__init__(**kwargs)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient

    def _make_one(self, x: np.ndarray) -> np.ndarray:
        # res = np.zeros_like(x)
        res = cv2.Canny(x, threshold1=self.threshold1, threshold2=self.threshold2,
                        apertureSize=self.aperture_size, L2gradient=self.l2_gradient)
        res = np.float32(res) / 255
        return res

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        frames = self.video[t]
        return np.array([self._make_one(frame) for frame in frames])

    @overrides
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        return np.uint8(255 * gray(x)[..., 0:3])
