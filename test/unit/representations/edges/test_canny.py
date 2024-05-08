import numpy as np
from vre.representations.edges.canny import Canny
from vre.utils import FakeVideo

def test_canny_1():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    canny_repr = Canny(name="canny", dependencies=[], threshold1=100, threshold2=200, aperture_size=3, l2_gradient=True)

    frames = np.array(video[0:1])
    y_canny = canny_repr(frames)
    assert y_canny.shape == (1, 64, 128)
    y_canny_images = canny_repr.make_images(slice(0, 1), y_canny)
    assert y_canny_images.shape == (1, 64, 128, 3)
    assert y_canny_images.dtype == np.uint8, y_canny_images.dtype

    assert canny_repr.size(y_canny) == (64, 128)
    y_canny_resized = canny_repr.resize(y_canny, (32, 64)) # we can resize it though
    assert canny_repr.size(y_canny_resized) == (32, 64)
    assert canny_repr.make_images(frames, y_canny_resized).shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_canny_1()
