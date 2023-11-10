import numpy as np
from vre.representations.depth.dpt import DepthDpt
from vre.utils import FakeVideo

def test_dpt():
    rgb_data = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), 30)
    dpt_repr = DepthDpt(video=rgb_data, name="dpt", dependencies=[])
    y_dpt, extra = dpt_repr(slice(0, 1))
    assert y_dpt.shape == (1, 192, 384), y_dpt.shape
    assert extra == {}, extra
    y_dpt_images = dpt_repr.make_images(slice(0, 1), y_dpt, extra)
    assert y_dpt_images.shape == (1, 64, 128, 3), y_dpt_images.shape
    assert y_dpt_images.dtype == np.uint8, y_dpt_images.dtype

if __name__ == "__main__":
    test_dpt()
