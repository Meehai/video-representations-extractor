import numpy as np
from vre.representations.rgb import RGB
from vre.utils import FakeVideo

def test_rgb_1():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    rgb_repr = RGB("rgb")

    frames = np.array(video[0:1])
    y_rgb = rgb_repr(frames)
    assert np.allclose(y_rgb.output, video.data[0])

    y_rgb_images = rgb_repr.make_images(frames, y_rgb)
    assert np.allclose(y_rgb_images, video.data[0])
    assert y_rgb_images.shape == (1, 64, 128, 3) and y_rgb_images.dtype == np.uint8, y_rgb_images.dtype

def test_rgb_resize():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    rgb_repr = RGB("rgb")

    frames = np.array(video[0:1])
    y_rgb = rgb_repr(frames)
    assert rgb_repr.size(y_rgb) == (64, 128)

    y_rgb_resized = rgb_repr.resize(y_rgb, (32, 64))
    assert rgb_repr.size(y_rgb_resized) == (32, 64)

    y_rgb_images_resized = rgb_repr.make_images(frames, y_rgb_resized)
    assert y_rgb_images_resized.shape == (1, 32, 64, 3) and y_rgb_images_resized.dtype == np.uint8
