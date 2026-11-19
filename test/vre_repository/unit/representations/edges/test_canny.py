import numpy as np
import pytest
from vre_repository.edges.canny import Canny, canny
from vre_video import VREVideo

def test_canny_1():
    video = VREVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
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

def test_canny_step_edge():
    """a 0 -> 255 step has a 2-pixel wide gradient plateau; the edge is a 1-pixel line on its left (top) pixel"""
    image = np.zeros((8, 8), dtype=np.uint8)
    image[:, 4:] = 255
    expected = np.zeros((8, 8), dtype=bool)
    expected[:, 3] = True
    assert np.array_equal(canny(image, 100, 200), expected)
    assert np.array_equal(canny(image.T.copy(), 100, 200), expected.T)
    assert np.array_equal(canny(np.stack([image, image, image], axis=-1), 100, 200), expected) # RGB: same edges

def test_canny_hysteresis_drops_isolated_weak_edges():
    """two steps: 0 -> 200 (gradient 800, strong) and 200 -> 240 (gradient 160, weak unless high < 160)"""
    image = np.zeros((8, 16), dtype=np.uint8)
    image[:, 4:] = 200
    image[:, 12:] = 240
    strong_only = np.zeros((8, 16), dtype=bool)
    strong_only[:, 3] = True
    both = strong_only.copy()
    both[:, 11] = True
    assert np.array_equal(canny(image, 100, 500), strong_only)
    assert np.array_equal(canny(image, 500, 100), strong_only) # threshold order does not matter
    assert np.array_equal(canny(image, 100, 150), both)

def test_canny_bad_inputs():
    with pytest.raises(TypeError):
        canny(np.zeros((8, 8), dtype=np.float32), 100, 200)
    with pytest.raises(ValueError):
        canny(np.zeros((8, 8), dtype=np.uint8), 100, 200, aperture_size=4)

if __name__ == "__main__":
    test_canny_1()
