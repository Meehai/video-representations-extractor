import numpy as np
from vre.representations.optical_flow.raft import FlowRaft

def test_raft():
    rgb_data = np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8)
    iters = np.random.randint(2, 10)
    small = np.random.choice([True, False])
    raft_repr = FlowRaft(video=rgb_data, name="raft", dependencies=[], inference_height=128, inference_width=128,
                         iters=iters, small=small, mixed_precision=False)
    y_raft, extra = raft_repr(slice(0, 1))
    assert y_raft.shape == (1, 128, 128, 2), (y_raft.shape, iters, small)
    assert extra == {}, (extra, iters, small)

    y_raft_images = raft_repr.make_images(y_raft, extra)
    assert y_raft_images.shape == (1, 128, 128, 3), y_raft_images.shape
    assert y_raft_images.dtype == np.uint8, y_raft_images.dtype

if __name__ == "__main__":
    test_raft()
