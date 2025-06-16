import numpy as np
from vre_repository.edges.canny import Canny
from vre import FrameVideo

def test_canny_1():
    video = FrameVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    canny_repr = Canny(name="canny", dependencies=[], threshold1=100, threshold2=200, aperture_size=3, l2_gradient=True)
    assert canny_repr.name == "canny"
    assert canny_repr.compress is True # default

    out = canny_repr.compute(video, ixs=[0])
    assert out.output.shape == (1, 64, 128, 1)
    out_images = canny_repr.make_images(out)
    assert out_images.shape == (1, 64, 128, 3)
    assert out_images.dtype == np.uint8, out_images.dtype

    assert canny_repr.size(out) == (1, 64, 128, 1)
    out_resized = canny_repr.resize(out, (32, 64)) # we can resize it though
    assert canny_repr.size(out_resized) == (1, 32, 64, 1)
    assert canny_repr.make_images(out_resized).shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_canny_1()
