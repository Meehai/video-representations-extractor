"""HSV module"""
import numpy as np
from overrides import overrides

from vre_video import VREVideo
from vre.utils import ReprOut

from .rgb import RGB
from .color_representation import ColorRepresentation

def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB to HSV color space conversion."""
    assert rgb.ndim in (3, 4), f"Expected 3 or 4 dimensions, got {rgb.ndim}"

    arr = rgb.astype(np.float64) / 255 if rgb.dtype == np.uint8 else rgb
    out = np.empty_like(arr)

    # -- V channel
    out_v = arr.max(-1)

    # -- S channel
    delta = np.ptp(arr, axis=-1)
    # Ignore warning for zero divided by zero
    old_settings = np.seterr(invalid='ignore')
    out_s = delta / out_v
    out_s[delta == 0.0] = 0.0

    # -- H channel
    # red is max
    idx = arr[..., 0] == out_v
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]

    # green is max
    idx = arr[..., 1] == out_v
    out[idx, 0] = 2.0 + (arr[idx, 2] - arr[idx, 0]) / delta[idx]

    # blue is max
    idx = arr[..., 2] == out_v
    out[idx, 0] = 4.0 + (arr[idx, 0] - arr[idx, 1]) / delta[idx]
    out_h = (out[..., 0] / 6.0) % 1.0
    out_h[delta == 0.0] = 0.0

    np.seterr(**old_settings)

    # -- output
    out[..., 0] = out_h
    out[..., 1] = out_s
    out[..., 2] = out_v

    # remove NaN
    out[np.isnan(out)] = 0

    return out

class HSV(ColorRepresentation):
    """HSV representation"""
    def __init__(self, name: str, dependencies: list[str]):
        assert len(dependencies) == 1 and isinstance(dependencies[0], RGB), dependencies
        ColorRepresentation.__init__(self, name, dependencies=dependencies)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        assert dep_data is not None and len(dep_data) == 1, dep_data
        return ReprOut(frames=video[ixs], output=rgb2hsv(dep_data[0].output).astype(np.float32), key=ixs)

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        y = self.unnormalize(data.output) if self.normalization is not None else data.output
        return (y * 255).astype(np.uint8)
