import numpy as np
from vre.representations.soft_segmentation.generalized_boundaries import GeneralizedBoundaries
from vre.utils import FakeVideo

def test_generalized_boundaries():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    gb_repr = GeneralizedBoundaries(name="gb", dependencies=[], use_median_filtering=True,
                                    adjust_to_rgb=True, max_channels=3)

    frames = video[0:1]
    y_gb = gb_repr(frames)
    assert y_gb.shape == (1, 64, 128, 3), y_gb.shape
    y_gb_images = gb_repr.make_images(frames, y_gb)
    assert y_gb_images.shape == (1, 64, 128, 3), y_gb_images.shape
    assert y_gb_images.dtype == np.uint8, y_gb_images.dtype

if __name__ == "__main__":
    test_generalized_boundaries()
