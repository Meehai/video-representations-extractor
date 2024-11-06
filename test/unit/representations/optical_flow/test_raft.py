import numpy as np
from vre.representations.optical_flow.raft import FlowRaft
from vre.utils import FakeVideo

def test_raft():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 255, 255, 3), dtype=np.uint8), frame_rate=30)
    iters = np.random.randint(2, 5)
    small = bool(np.random.randint(0, 2))
    raft_repr = FlowRaft(name="raft", dependencies=[], inference_height=128, inference_width=128,
                         iters=iters, small=small)
    raft_repr.video = video
    raft_repr.vre_setup(load_weights=False)
    assert raft_repr.name == "raft"
    assert raft_repr.compress is True # default from ComputeRepresentationMixin
    assert raft_repr.device == "cpu" # default from LearnedRepresentationMixin

    frames = np.array(video[0:1])
    right_frames = np.array(video[1:2])
    y_raft = raft_repr.make(frames, {"right_frames": right_frames})
    assert y_raft.output.shape == (1, 128, 128, 2), (y_raft.output.shape, iters)

    y_raft_images = raft_repr.make_images(frames, y_raft)
    assert y_raft_images.shape == (1, 128, 128, 3), y_raft_images.shape
    assert y_raft_images.dtype == np.uint8, y_raft_images.dtype

    assert raft_repr.size(y_raft) == (raft_repr.inference_height, raft_repr.inference_width)
    y_raft_resized = raft_repr.resize(y_raft, (64, 128)) # we can resize it though
    assert raft_repr.size(y_raft_resized) == (64, 128)
    assert raft_repr.make_images(frames, y_raft_resized).shape == (1, 64, 128, 3)
