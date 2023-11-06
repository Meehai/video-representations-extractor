import numpy as np
from vre.representations.soft_segmentation.halftone import Halftone

def test_halftone():
    rgb_data = np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8)
    halftone_repr = Halftone(video=rgb_data, name="halftone", dependencies=[], sample=3, scale=1, percentage=91,
                             angles=[0, 15, 30, 45], antialias=False, resolution=rgb_data.shape[1:3])
    y_halftone, extra = halftone_repr(slice(0, 1))
    assert y_halftone.shape == (1, 64, 128, 3), y_halftone.shape
    assert extra == {}, extra
    y_halftone_images = halftone_repr.make_images(y_halftone, extra)
    assert y_halftone_images.shape == (1, 64, 128, 3), y_halftone_images.shape
    assert y_halftone_images.dtype == np.uint8, y_halftone_images.dtype

if __name__ == "__main__":
    test_halftone()
