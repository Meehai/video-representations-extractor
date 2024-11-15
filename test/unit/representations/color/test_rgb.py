import numpy as np
from vre.representations.color import RGB
from vre.utils import FakeVideo

def test_rgb_1():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    rgb_repr = RGB("rgb")
    assert rgb_repr.name == "rgb"
    assert rgb_repr.compress is True # default from NpIORepresentation
    assert rgb_repr.batch_size == 1 # defasult from ComputeRepreserntation

    rgb_repr.compute(video, [0])
    assert np.allclose(rgb_repr.data.output, video.data[0])

    y_rgb_images = rgb_repr.make_images()
    assert np.allclose(y_rgb_images, video.data[0])
    assert y_rgb_images.shape == (1, 64, 128, 3) and y_rgb_images.dtype == np.uint8, y_rgb_images.dtype

def test_rgb_resize():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    rgb_repr = RGB("rgb")

    rgb_repr.compute(video, [0])
    assert rgb_repr.size == (1, 64, 128, 3)

    rgb_repr.data = rgb_repr.resize(rgb_repr.data, (32, 64))
    assert rgb_repr.size == (1, 32, 64, 3)

    y_rgb_images_resized = rgb_repr.make_images()
    assert y_rgb_images_resized.shape == (1, 32, 64, 3) and y_rgb_images_resized.dtype == np.uint8
