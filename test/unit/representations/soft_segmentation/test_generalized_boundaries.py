import numpy as np
from vre.representations.soft_segmentation.generalized_boundaries import GeneralizedBoundaries
from vre.utils import FakeVideo

def test_generalized_boundaries():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    gb_repr = GeneralizedBoundaries(name="gb", dependencies=[], use_median_filtering=True,
                                    adjust_to_rgb=True, max_channels=3)
    assert gb_repr.name == "gb"
    assert gb_repr.compress is True # default

    frames = video[0:1]
    gb_repr.compute(video, [0])
    assert gb_repr.data.output.shape == (1, 64, 128, 3), gb_repr.data.output.shape
    y_gb_images = gb_repr.make_images()
    assert y_gb_images.shape == (1, 64, 128, 3), y_gb_images.shape
    assert y_gb_images.dtype == np.uint8, y_gb_images.dtype

    assert gb_repr.size == (1, 64, 128, 3)
    gb_repr.data = gb_repr.resize(gb_repr.data, (32, 64))
    assert gb_repr.size == (1, 32, 64, 3)
    assert gb_repr.make_images().shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_generalized_boundaries()
