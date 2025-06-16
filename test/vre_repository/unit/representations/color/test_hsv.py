import numpy as np
from vre import FrameVideo
from vre_repository.color.rgb import RGB
from vre_repository.color.hsv import HSV

def test_hsv_compute():
    video = FrameVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    hsv_repr = HSV("hsv", dependencies=[RGB("rgb")])
    assert hsv_repr.name == "hsv"
    assert hsv_repr.compress is True # default

    rgb_out = hsv_repr.dependencies[0].compute(video, [0])
    out = hsv_repr.compute(video, [0], [rgb_out])
    assert out.output.shape == (1, 64, 128, 3)

    out_images = hsv_repr.make_images(out)
    assert out_images.shape == (1, 64, 128, 3)
    assert out_images.dtype == np.uint8

def test_hsv_resize():
    video = FrameVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    hsv_repr = HSV("hsv", dependencies=[RGB("rgb")])

    rgb_out = hsv_repr.dependencies[0].compute(video, [0])
    out = hsv_repr.compute(video, [0], [rgb_out])
    assert hsv_repr.size(out) == (1, 64, 128, 3)

    out_resized = hsv_repr.resize(out, (32, 64))
    assert hsv_repr.size(out_resized) == (1, 32, 64, 3)

    out_images_resized = hsv_repr.make_images(out_resized)
    assert out_images_resized.shape == (1, 32, 64, 3) and out_images_resized.dtype == np.uint8
