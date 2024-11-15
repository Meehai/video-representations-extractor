import numpy as np
from vre.representations.soft_segmentation.halftone import Halftone
from vre.utils import FakeVideo

def test_halftone():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    halftone_repr = Halftone(name="halftone", dependencies=[], sample=3, scale=1, percentage=91,
                             angles=[0, 15, 30, 45], antialias=False, resolution=video.frame_shape[0:2])
    assert halftone_repr.name == "halftone"
    assert halftone_repr.compress is True # default from IORepresentationMixin

    halftone_repr.compute(video, [0])
    assert halftone_repr.data.output.shape == (1, 64, 128, 3), halftone_repr.data.output.shape
    y_halftone_images = halftone_repr.make_images()
    assert y_halftone_images.shape == (1, 64, 128, 3), y_halftone_images.shape
    assert y_halftone_images.dtype == np.uint8, y_halftone_images.dtype

    assert halftone_repr.size == (1, 64, 128, 3)
    halftone_repr.data = halftone_repr.resize(halftone_repr.data, (32, 64))
    assert halftone_repr.size == (1, 32, 64, 3)
    assert halftone_repr.make_images().shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_halftone()
