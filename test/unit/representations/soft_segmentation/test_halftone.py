import numpy as np
from vre.representations.soft_segmentation.halftone import Halftone
from vre.utils import FakeVideo

def test_halftone():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    halftone_repr = Halftone(name="halftone", dependencies=[], sample=3, scale=1, percentage=91,
                             angles=[0, 15, 30, 45], antialias=False, resolution=video.frame_shape[0:2])

    frames = np.array(video[0:1])
    y_halftone = halftone_repr(frames)
    assert y_halftone.shape == (1, 64, 128, 3), y_halftone.shape
    y_halftone_images = halftone_repr.make_images(frames, y_halftone)
    assert y_halftone_images.shape == (1, 64, 128, 3), y_halftone_images.shape
    assert y_halftone_images.dtype == np.uint8, y_halftone_images.dtype

if __name__ == "__main__":
    test_halftone()
