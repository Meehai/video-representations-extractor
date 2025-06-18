import numpy as np
import pytest
from vre_repository.optical_flow.raft import FlowRaft
from vre_video import VREVideo

@pytest.mark.parametrize(["iters", "small"], [(2, False), (2, True), (4, True)])
def test_raft(iters: int, small: bool):
    video = VREVideo(np.random.randint(0, 255, size=(20, 255, 255, 3), dtype=np.uint8), fps=30)
    raft_repr = FlowRaft(name="raft", dependencies=[], inference_height=128, inference_width=128,
                         iters=iters, small=small, delta=1)
    raft_repr.vre_setup(load_weights=False)
    assert raft_repr.name == "raft"
    assert raft_repr.compress is True # default
    assert raft_repr.device == "cpu" # default from LearnedRepresentationMixin

    out = raft_repr.compute(video, [0])
    assert out.output.shape == (1, 128, 128, 2), (out.output.shape, iters)

    out_images = raft_repr.make_images(out)
    assert out_images.shape == (1, 128, 128, 3), out_images.shape
    assert out_images.dtype == np.uint8, out_images.dtype

    assert raft_repr.size(out) == (1, raft_repr.inference_height, raft_repr.inference_width, 2)
    out_resized = raft_repr.resize(out, (64, 128)) # we can resize it though
    assert raft_repr.size(out_resized) == (1, 64, 128, 2)
    assert raft_repr.make_images(out_resized).shape == (1, 64, 128, 3)
