import numpy as np
from vre.representations.soft_segmentation.generalized_boundaries import GeneralizedBoundaries
from vre.utils import FakeVideo

def test_generalized_boundaries():
    rgb_data = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), 30)
    gb_repr = GeneralizedBoundaries(video=rgb_data, name="dpt", dependencies=[], use_median_filtering=True,
                                    adjust_to_rgb=True, max_channels=3)
    y_gb, extra = gb_repr(slice(0, 1))
    assert y_gb.shape == (1, 64, 128, 3), y_gb.shape
    assert extra == {}, extra
    y_gb_images = gb_repr.make_images(slice(0, 1), y_gb, extra)
    assert y_gb_images.shape == (1, 64, 128, 3), y_gb_images.shape
    assert y_gb_images.dtype == np.uint8, y_gb_images.dtype

if __name__ == "__main__":
    test_generalized_boundaries()
