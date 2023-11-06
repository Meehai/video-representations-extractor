import numpy as np
from vre.representations.optical_flow.rife import FlowRife

def test_rife_uhd_false():
    rgb_data = np.random.randint(0, 255, size=(1, 64, 128, 3), dtype=np.uint8)
    rife_repr = FlowRife(video=rgb_data, name="rife", dependencies=[], compute_backward_flow=False, uhd=False)
    y_rife, extra = rife_repr(slice(0, 1))
    assert y_rife.shape == (1, 32, 64, 2), y_rife.shape
    assert extra == {}, extra

    y_rife_images = rife_repr.make_images(y_rife, extra)
    assert y_rife_images.shape == (1, 32, 64, 3), y_rife_images.shape
    assert y_rife_images.dtype == np.uint8, y_rife_images.dtype

def test_rife_uhd_true():
    rgb_data = np.random.randint(0, 255, size=(1, 64, 128, 3), dtype=np.uint8)
    rife_repr = FlowRife(video=rgb_data, name="rife", dependencies=[], compute_backward_flow=False, uhd=True)
    y_rife, extra = rife_repr(slice(0, 1))
    assert y_rife.shape == (1, 16, 32, 2), y_rife.shape
    assert extra == {}, extra

    y_rife_images = rife_repr.make_images(y_rife, extra)
    assert y_rife_images.shape == (1, 16, 32, 3), y_rife_images.shape
    assert y_rife_images.dtype == np.uint8, y_rife_images.dtype

if __name__ == "__main__":
    test_rife_uhd_false()
