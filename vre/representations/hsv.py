"""HSV module"""
import numpy as np
from overrides import overrides

from vre.utils import image_resize_batch
from vre.representations import Representation, ReprOut, ComputeRepresentationMixin

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

class HSV(Representation, ComputeRepresentationMixin):
    """HSV representation"""
    def __init__(self, *args, **kwargs):
        Representation.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        assert len(self.dependencies) == 0, self.dependencies

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, ReprOut] | None = None) -> ReprOut:
        return ReprOut(output=rgb2hsv(frames).astype(np.float32))

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: ReprOut) -> np.ndarray:
        return (repr_data.output * 255).astype(np.uint8)

    @overrides
    def size(self, repr_data: ReprOut) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: ReprOut, new_size: tuple[int, int]) -> ReprOut:
        return ReprOut(output=image_resize_batch(repr_data.output, height=new_size[0], width=new_size[1]))
