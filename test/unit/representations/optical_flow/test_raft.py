import numpy as np
import pytest
from vre.representations.optical_flow.raft import FlowRaft
from vre.utils import FakeVideo

@pytest.mark.parametrize(["iters", "small"], [(2, False), (2, True), (4, True)])
def test_raft(iters: int, small: bool):
    video = FakeVideo(np.random.randint(0, 255, size=(20, 255, 255, 3), dtype=np.uint8), fps=30)
    raft_repr = FlowRaft(name="raft", dependencies=[], inference_height=128, inference_width=128,
                         iters=iters, small=small)
    raft_repr.vre_setup(load_weights=False)
    assert raft_repr.name == "raft"
    assert raft_repr.compress is True # default from ComputeRepresentationMixin
    assert raft_repr.device == "cpu" # default from LearnedRepresentationMixin

    raft_repr.compute(video, [0])
    assert raft_repr.data.output.shape == (1, 128, 128, 2), (raft_repr.data.output.shape, iters)

    y_raft_images = raft_repr.make_images()
    assert y_raft_images.shape == (1, 128, 128, 3), y_raft_images.shape
    assert y_raft_images.dtype == np.uint8, y_raft_images.dtype

    assert raft_repr.size == (1, raft_repr.inference_height, raft_repr.inference_width, 2)
    raft_repr.data = raft_repr.resize(raft_repr.data, (64, 128)) # we can resize it though
    assert raft_repr.size == (1, 64, 128, 2)
    assert raft_repr.make_images().shape == (1, 64, 128, 3)
