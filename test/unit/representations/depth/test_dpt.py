import numpy as np
from vre.representations.depth.dpt import DepthDpt
from vre.utils import FakeVideo

def test_dpt():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    dpt_repr = DepthDpt(name="dpt", dependencies=[])

    frames = np.array(video[0:1])
    y_dpt = dpt_repr(frames)
    assert y_dpt.shape == (1, 192, 384), y_dpt.shape

    y_dpt_images = dpt_repr.make_images(frames, y_dpt)
    assert y_dpt_images.shape == (1, 64, 128, 3), y_dpt_images.shape
    assert y_dpt_images.dtype == np.uint8, y_dpt_images.dtype

if __name__ == "__main__":
    test_dpt()
