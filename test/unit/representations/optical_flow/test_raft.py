import numpy as np
from vre.representations.optical_flow.raft import FlowRaft
from vre.utils import FakeVideo

def test_raft_small_false():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    iters = np.random.randint(2, 5)
    raft_repr = FlowRaft(name="raft", dependencies=[], inference_height=128, inference_width=128,
                         iters=iters, small=False, mixed_precision=False)

    frames = np.array(video[0:1])
    right_frames = np.array(video[1:2])
    y_raft = raft_repr.make(frames, right_frames)
    assert y_raft.shape == (1, 128, 128, 2), (y_raft.shape, iters)

    y_raft_images = raft_repr.make_images(frames, y_raft)
    assert y_raft_images.shape == (1, 128, 128, 3), y_raft_images.shape
    assert y_raft_images.dtype == np.uint8, y_raft_images.dtype

def test_raft_small_true():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    iters = np.random.randint(2, 5)
    raft_repr = FlowRaft(name="raft", dependencies=[], inference_height=128, inference_width=128,
                         iters=iters, small=True, mixed_precision=False)

    frames = np.array(video[0:1])
    right_frames = np.array(video[1:2])
    y_raft = raft_repr.make(frames, right_frames)
    assert y_raft.shape == (1, 128, 128, 2), (y_raft.shape, iters)

    y_raft_images = raft_repr.make_images(frames, y_raft)
    assert y_raft_images.shape == (1, 128, 128, 3), y_raft_images.shape
    assert y_raft_images.dtype == np.uint8, y_raft_images.dtype

if __name__ == "__main__":
    test_raft_small_false()
