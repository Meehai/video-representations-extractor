import numpy as np
from vre_repository.color.rgb import RGB
from vre import FrameVideo

def test_rgb_1():
    video = FrameVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    rgb_repr = RGB("rgb")
    assert rgb_repr.name == "rgb"
    assert rgb_repr.compress is True # default from NpIORepresentation
    assert rgb_repr.batch_size == 1 # deefault

    out = rgb_repr.compute(video, [0])
    assert np.allclose(out.output, video.data[0])

    out_images = rgb_repr.make_images(out)
    assert np.allclose(out_images, video.data[0])
    assert out_images.shape == (1, 64, 128, 3) and out_images.dtype == np.uint8, out_images.dtype

def test_rgb_resize():
    video = FrameVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    rgb_repr = RGB("rgb")

    out = rgb_repr.compute(video, [0])
    assert rgb_repr.size(out) == (1, 64, 128, 3)

    out_resized = rgb_repr.resize(out, (32, 64))
    assert rgb_repr.size(out_resized) == (1, 32, 64, 3)

    out_images_resized = rgb_repr.make_images(out_resized)
    assert out_images_resized.shape == (1, 32, 64, 3) and out_images_resized.dtype == np.uint8
