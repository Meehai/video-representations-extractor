"""Canny edge detector representation."""
import numpy as np
from overrides import overrides
from matplotlib.cm import gray # pylint: disable=no-name-in-module

from vre.representations import Representation, ReprOut, ComputeRepresentationMixin, NpIORepresentation
from vre.utils import VREVideo, MemoryData
from vre.utils.cv2_utils import cv2_Canny

class Canny(Representation, ComputeRepresentationMixin, NpIORepresentation):
    """Canny edge detector representation."""
    def __init__(self, threshold1: float, threshold2: float, aperture_size: int, l2_gradient: bool, **kwargs):
        Representation.__init__(self, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        self.data = ReprOut(frames=video[ixs], key=ixs,
                            output=MemoryData([self._make_one(frame) for frame in video[ixs]]))

    @overrides
    def make_images(self) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        return (255 * gray(self.data.output)[..., 0:3]).astype(np.uint8)

    def _make_one(self, x: np.ndarray) -> np.ndarray:
        res = cv2_Canny(x, threshold1=self.threshold1, threshold2=self.threshold2,
                        apertureSize=self.aperture_size, L2gradient=self.l2_gradient)
        res = np.float32(res) / 255
        return res
