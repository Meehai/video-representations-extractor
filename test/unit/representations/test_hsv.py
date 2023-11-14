import numpy as np
from vre.representations.hsv import HSV
from vre.utils import FakeVideo

def test_rgb_1():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    hsv_repr = HSV("hsv", [])

    frames = np.array(video[0:1])
    y_hsv = hsv_repr(frames)
    assert y_hsv.shape == (1, 64, 128, 3), y_hsv.shape

    y_hsv_images = hsv_repr.make_images(frames, y_hsv)
    assert y_hsv_images.shape == (1, 64, 128, 3), y_hsv_images.shape
    assert y_hsv_images.dtype == np.uint8, y_hsv_images.dtype
