import numpy as np
from vre.representations.edges.canny import Canny
from vre.utils import FakeVideo

def test_canny_1():
    rgb_data = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), 30)
    canny_repr = Canny(video=rgb_data, name="canny", dependencies=[], threshold1=100, threshold2=200,
                       aperture_size=3, l2_gradient=True)
    y_canny, extra = canny_repr(slice(0, 1))
    assert y_canny.shape == (1, 64, 128)
    assert extra == {}
    y_canny_images = canny_repr.make_images(slice(0, 1), y_canny, extra)
    assert y_canny_images.shape == (1, 64, 128, 3)
    assert y_canny_images.dtype == np.uint8, y_canny_images.dtype

if __name__ == "__main__":
    test_canny_1()
