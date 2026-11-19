"""Canny edge detector representation. `canny` is a numpy port of cv2.Canny: same edge pixels."""
import math
import numpy as np
from overrides import overrides

from vre_video import VREVideo
from vre.utils import MemoryData
from vre import ReprOut

from ..edges import EdgesRepresentation

# (smoothing, derivative) 1D Sobel kernels per aperture size, as cv2.getDerivKernels builds them
_SOBEL_KERNELS = {3: ([1, 2, 1], [-1, 0, 1]),
                  5: ([1, 4, 6, 4, 1], [-1, -2, 0, 2, 1]),
                  7: ([1, 6, 15, 20, 15, 6, 1], [-1, -4, -5, 0, 5, 4, 1])}
_TG22 = 13573 # tan(22.5 deg) * 2^15: cv2's fixed-point gradient direction test

class Canny(EdgesRepresentation):
    """Canny edge detector representation."""
    def __init__(self, threshold1: float, threshold2: float, aperture_size: int, l2_gradient: bool, **kwargs):
        super().__init__(**kwargs)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient

    @overrides
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        output = MemoryData([self._make_one(frame) for frame in video[ixs]])
        return ReprOut(frames=video[ixs], key=ixs, output=output)

    def _make_one(self, x: np.ndarray) -> np.ndarray:
        res = canny(x, self.threshold1, self.threshold2, self.aperture_size, self.l2_gradient)
        return np.float32(res)[..., None]

def canny(image: np.ndarray, threshold1: float, threshold2: float, aperture_size: int = 3,
          l2_gradient: bool = False) -> np.ndarray:
    """cv2.Canny in numpy. image: (H, W) or (H, W, C) uint8. Returns the (H, W) bool edge mask.
    Multi-channel images use, per pixel, the channel with the strongest gradient (like cv2)."""
    if image.dtype != np.uint8:
        raise TypeError(f"Canny expects an uint8 image, got {image.dtype}")
    if aperture_size not in _SOBEL_KERNELS:
        raise ValueError(f"aperture_size must be one of {list(_SOBEL_KERNELS)}, got {aperture_size}")
    low, high = _thresholds(threshold1, threshold2, aperture_size, l2_gradient)
    dx, dy = _sobel(np.atleast_3d(image), aperture_size)
    mag = dx * dx + dy * dy if l2_gradient else np.abs(dx) + np.abs(dy)
    channel = mag.argmax(axis=2)[..., None] # strongest channel per pixel, the first one on ties
    mag, dx, dy = (np.take_along_axis(x, channel, axis=2)[..., 0] for x in (mag, dx, dy))
    candidates = (mag > low) & _non_max_suppression(mag, dx, dy)
    return _hysteresis(candidates & (mag > high), candidates)

def _thresholds(threshold1: float, threshold2: float, aperture_size: int, l2_gradient: bool) -> tuple[int, int]:
    """(low, high) integer thresholds on the gradient magnitude, derived like cv2.Canny does"""
    low, high = sorted((threshold1, threshold2))
    if aperture_size == 7: # the 7x7 Sobel is scaled by 1/16 (see _sobel), so are the thresholds
        low, high = low / 16, high / 16
    if l2_gradient: # the L2 magnitude is dx^2 + dy^2 (no sqrt), so the thresholds are squared
        low, high = (min(t, 32767) * min(t, 32767) if t > 0 else t for t in (low, high))
    return math.floor(low), math.floor(high)

def _sobel(image: np.ndarray, aperture_size: int) -> tuple[np.ndarray, np.ndarray]:
    """(dx, dy) int32 of an (H, W, C) uint8 image: cv2.Sobel(image, CV_16S, ...) with BORDER_REPLICATE"""
    smooth, deriv = _SOBEL_KERNELS[aperture_size]
    h, w, r = image.shape[0], image.shape[1], aperture_size // 2
    padded = np.pad(image.astype(np.int32), ((r, r), (r, r), (0, 0)), mode="edge")
    smooth_x = sum(k * padded[:, i:i + w] for i, k in enumerate(smooth)) # (H + 2r, W, C)
    deriv_x = sum(k * padded[:, i:i + w] for i, k in enumerate(deriv))
    dx = sum(k * deriv_x[i:i + h] for i, k in enumerate(smooth))
    dy = sum(k * smooth_x[i:i + h] for i, k in enumerate(deriv))
    if aperture_size == 7: # cv2 scales by 1/16 to fit int16 and rounds half to even (like np.rint)
        dx, dy = np.rint(dx / 16).astype(np.int32), np.rint(dy / 16).astype(np.int32)
    return dx, dy

def _non_max_suppression(mag: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """True where mag is a local max along the gradient direction, quantized to 0/45/90/135 degrees.
    Ties: strict on the left/top side, non-strict on the right/bottom, strict on both diagonal sides (as cv2)."""
    p = np.pad(mag, 1) # 0 magnitude outside the image
    x, y = np.abs(dx), np.abs(dy) << 15 # Q15 fixed point: y/x vs tan(22.5) and tan(67.5) in integers
    tg22x = x * _TG22
    tg67x = tg22x + (x << 16)
    left_right = (mag > p[1:-1, :-2]) & (mag >= p[1:-1, 2:])
    top_bottom = (mag > p[:-2, 1:-1]) & (mag >= p[2:, 1:-1])
    diagonal = np.where((dx ^ dy) >= 0, (mag > p[:-2, :-2]) & (mag > p[2:, 2:]), # same signs: top-left/bottom-right
                        (mag > p[:-2, 2:]) & (mag > p[2:, :-2]))                    # else: top-right/bottom-left
    return np.where(y < tg22x, left_right, np.where(y > tg67x, top_bottom, diagonal))

def _hysteresis(strong: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """The candidates 8-connected to a strong pixel (strong is a subset of candidates). Flood fill, one ring a round."""
    h, w = strong.shape
    is_candidate = np.pad(candidates, 1).ravel() # False ring: neighbours of a pixel never leave the array
    edges = np.pad(strong, 1).ravel()
    step = w + 2
    neighbours = np.array([-step - 1, -step, -step + 1, -1, 1, step - 1, step, step + 1])
    frontier = np.flatnonzero(edges)
    for _ in range(int(candidates.sum())): # every round adds >= 1 pixel, so this bound is never reached
        if frontier.size == 0:
            break
        ring = np.unique(frontier[:, None] + neighbours)
        frontier = ring[is_candidate[ring] & ~edges[ring]]
        edges[frontier] = True
    assert frontier.size == 0, f"hysteresis did not converge: {frontier.size} pixels left"
    return edges.reshape(h + 2, w + 2)[1:-1, 1:-1]
