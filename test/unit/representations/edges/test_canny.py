import numpy as np
from vre.representations.edges.canny import Canny
from vre.utils import FakeVideo

def test_canny_1():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    canny_repr = Canny(name="canny", dependencies=[], threshold1=100, threshold2=200, aperture_size=3, l2_gradient=True)
    assert canny_repr.name == "canny"
    assert canny_repr.compress is True # default from ComputeRepresentationMixin

    canny_repr.compute(video, ixs=[0])
    assert canny_repr.data.output.shape == (1, 64, 128)
    y_canny_images = canny_repr.make_images()
    assert y_canny_images.shape == (1, 64, 128, 3)
    assert y_canny_images.dtype == np.uint8, y_canny_images.dtype

    assert canny_repr.size == (1, 64, 128)
    canny_repr.data = canny_repr.resize(canny_repr.data, (32, 64)) # we can resize it though
    assert canny_repr.size == (1, 32, 64)
    assert canny_repr.make_images().shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_canny_1()
