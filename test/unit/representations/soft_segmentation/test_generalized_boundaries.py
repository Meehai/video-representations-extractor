import numpy as np
from vre.representations.soft_segmentation.generalized_boundaries import GeneralizedBoundaries
from vre.utils import FakeVideo

def test_generalized_boundaries():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    gb_repr = GeneralizedBoundaries(name="gb", dependencies=[], use_median_filtering=True,
                                    adjust_to_rgb=True, max_channels=3)

    frames = video[0:1]
    y_gb = gb_repr(frames)
    assert y_gb.output.shape == (1, 64, 128, 3), y_gb.output.shape
    y_gb_images = gb_repr.make_images(frames, y_gb)
    assert y_gb_images.shape == (1, 64, 128, 3), y_gb_images.shape
    assert y_gb_images.dtype == np.uint8, y_gb_images.dtype

    assert gb_repr.size(y_gb) == (64, 128)
    y_gb_resized = gb_repr.resize(y_gb, (32, 64))
    assert gb_repr.size(y_gb_resized) == (32, 64)
    assert gb_repr.make_images(frames, y_gb_resized).shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_generalized_boundaries()
