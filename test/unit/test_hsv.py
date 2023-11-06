import numpy as np
from vre.representations.hsv import HSV

def test_rgb_1():
    rgb_data = np.random.randint(0, 255, size=(1, 64, 128, 3), dtype=np.uint8)
    hsv_repr = HSV(rgb_data, "hsv", [])
    y_hsv, extra = hsv_repr(slice(0, 1))
    assert y_hsv.shape == (1, 64, 128, 3), y_hsv.shape
    assert extra == {}, extra

    y_hsv_images = hsv_repr.make_images(y_hsv, extra)
    assert y_hsv_images.shape == (1, 64, 128, 3), y_hsv_images.shape
    assert y_hsv_images.dtype == np.uint8, y_hsv_images.dtype
