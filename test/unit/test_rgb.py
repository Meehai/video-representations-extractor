import numpy as np
from vre.representations.rgb import RGB

def test_rgb_1():
    rgb_data = np.random.randint(0, 255, size=(1, 64, 128, 3), dtype=np.uint8)
    rgb_repr = RGB(rgb_data, "rgb", [])
    y_rgb, extra = rgb_repr(slice(0, 1))
    assert np.allclose(y_rgb * 255, rgb_data)
    assert extra == {}, extra

    y_rgb_images = rgb_repr.make_images(y_rgb, extra)
    assert np.allclose(y_rgb_images, rgb_data)
    assert y_rgb_images.dtype == np.uint8, y_rgb_images.dtype
