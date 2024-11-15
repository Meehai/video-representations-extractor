import numpy as np
from vre.representations.color import RGB, HSV
from vre.utils import FakeVideo

def test_hsv_compute():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    hsv_repr = HSV("hsv", dependencies=[RGB("rgb")])
    assert hsv_repr.name == "hsv"
    assert hsv_repr.compress is True # default from ComputeRepresentationMixin

    hsv_repr.dependencies[0].compute(video, [0]) # TODO: can this be automated?
    hsv_repr.compute(video, [0])
    assert hsv_repr.data.output.shape == (1, 64, 128, 3), hsv_repr.data.output.shape

    y_hsv_images = hsv_repr.make_images()
    assert y_hsv_images.shape == (1, 64, 128, 3), y_hsv_images.shape
    assert y_hsv_images.dtype == np.uint8, y_hsv_images.dtype

def test_hsv_resize():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    hsv_repr = HSV("hsv", dependencies=[RGB("rgb")])

    hsv_repr.dependencies[0].compute(video, [0]) # TODO: can this be automated?
    hsv_repr.compute(video, ixs=[0])
    assert hsv_repr.size == (1, 64, 128, 3), hsv_repr.size

    hsv_repr.data = hsv_repr.resize(hsv_repr.data, (32, 64))
    assert hsv_repr.size == (1, 32, 64, 3), hsv_repr.size

    y_hsv_images_resized = hsv_repr.make_images()
    assert y_hsv_images_resized.shape == (1, 32, 64, 3) and y_hsv_images_resized.dtype == np.uint8
