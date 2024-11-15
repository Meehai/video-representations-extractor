import numpy as np
import pytest
from vre.representations.optical_flow.rife import FlowRife
from vre.utils import FakeVideo

@pytest.mark.parametrize("uhd", [False, True])
def test_rife_uhd_false(uhd):
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    rife_repr = FlowRife(name="rife", dependencies=[], compute_backward_flow=False, uhd=uhd)
    rife_repr.vre_setup(load_weights=False)
    assert rife_repr.name == "rife"
    assert rife_repr.compress is True # default from ComputeRepresentationMixin
    assert rife_repr.device == "cpu" # default from LearnedRepresentationMixin

    out_shape = (32, 64) if not uhd else (16, 32)
    rife_repr.compute(video, [0])
    assert rife_repr.data.output.shape == (1, *out_shape, 2), rife_repr.data.output.shape

    y_rife_images = rife_repr.make_images()
    assert y_rife_images.shape == (1, *out_shape, 3), y_rife_images.shape
    assert y_rife_images.dtype == np.uint8, y_rife_images.dtype

    assert rife_repr.size == (1, *out_shape, 2)
    rife_repr.data = rife_repr.resize(rife_repr.data, (64, 128)) # we can resize it though
    assert rife_repr.size == (1, 64, 128, 2)
    assert rife_repr.make_images().shape == (1, 64, 128, 3)
