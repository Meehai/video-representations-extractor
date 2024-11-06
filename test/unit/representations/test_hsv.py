import numpy as np
from vre.representations.hsv import HSV
from vre.utils import FakeVideo

def test_hsv_make():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    hsv_repr = HSV("hsv")
    assert hsv_repr.name == "hsv"
    assert hsv_repr.compress is True # default from ComputeRepresentationMixin

    frames = np.array(video[0:1])
    y_hsv = hsv_repr(frames)
    assert y_hsv.output.shape == (1, 64, 128, 3), y_hsv.output.shape

    y_hsv_images = hsv_repr.make_images(frames, y_hsv)
    assert y_hsv_images.shape == (1, 64, 128, 3), y_hsv_images.shape
    assert y_hsv_images.dtype == np.uint8, y_hsv_images.dtype

def test_hsv_resize():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    hsv_repr = HSV("hsv")

    frames = np.array(video[0:1])
    y_hsv = hsv_repr(frames)
    assert hsv_repr.size(y_hsv) == (64, 128)

    y_hsv_resized = hsv_repr.resize(y_hsv, (32, 64))
    assert hsv_repr.size(y_hsv_resized) == (32, 64)

    y_hsv_images_resized = hsv_repr.make_images(frames, y_hsv_resized)
    assert y_hsv_images_resized.shape == (1, 32, 64, 3) and y_hsv_images_resized.dtype == np.uint8
