import numpy as np
import pytest
from vre_repository.optical_flow.rife import FlowRife
from vre_video import VREVideo

@pytest.mark.parametrize("uhd", [False, True])
def test_rife_uhd_false(uhd):
    video = VREVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    rife_repr = FlowRife(name="rife", dependencies=[], compute_backward_flow=False, uhd=uhd, delta=1)
    rife_repr.vre_setup(load_weights=False)
    assert rife_repr.name == "rife"
    assert rife_repr.compress is True # default
    assert rife_repr.device == "cpu" # default from LearnedRepresentationMixin

    out_shape = (32, 64) if not uhd else (16, 32)
    out = rife_repr.compute(video, [0])
    assert out.output.shape == (1, *out_shape, 2), out.output.shape

    out_images = rife_repr.make_images(out)
    assert out_images.shape == (1, *out_shape, 3), out_images.shape
    assert out_images.dtype == np.uint8, out_images.dtype

    assert rife_repr.size(out) == (1, *out_shape, 2)
    out_resized = rife_repr.resize(out, (64, 128)) # we can resize it though
    assert rife_repr.size(out_resized) == (1, 64, 128, 2)
    assert rife_repr.make_images(out_resized).shape == (1, 64, 128, 3)
