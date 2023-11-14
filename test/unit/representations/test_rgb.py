import numpy as np
from vre.representations.rgb import RGB
from vre.utils import FakeVideo

def test_rgb_1():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    rgb_repr = RGB("rgb", [])

    frames = np.array(video[0:1])
    y_rgb = rgb_repr(frames)
    assert np.allclose(y_rgb * 255, video.data[0])

    y_rgb_images = rgb_repr.make_images(frames, y_rgb)
    assert np.allclose(y_rgb_images, video.data[0])
    assert y_rgb_images.dtype == np.uint8, y_rgb_images.dtype
