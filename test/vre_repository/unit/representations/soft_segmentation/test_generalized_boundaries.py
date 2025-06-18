import numpy as np
from vre_repository.soft_segmentation.generalized_boundaries import GeneralizedBoundaries
from vre_video import VREVideo

def test_generalized_boundaries():
    video = VREVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    gb_repr = GeneralizedBoundaries(name="gb", dependencies=[], use_median_filtering=True,
                                    adjust_to_rgb=True, max_channels=3)
    assert gb_repr.name == "gb"
    assert gb_repr.compress is True # default

    out = gb_repr.compute(video, [0])
    assert out.output.shape == (1, 64, 128, 3)
    out_images = gb_repr.make_images(out)
    assert out_images.shape == (1, 64, 128, 3)
    assert out_images.dtype == np.uint8

    assert gb_repr.size(out) == (1, 64, 128, 3)
    out_resized = gb_repr.resize(out, (32, 64))
    assert gb_repr.size(out_resized) == (1, 32, 64, 3)
    assert gb_repr.make_images(out_resized).shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_generalized_boundaries()
