import numpy as np
from vre.representations.optical_flow.rife import FlowRife
from vre.utils import FakeVideo

def test_rife_uhd_false():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    rife_repr = FlowRife(name="rife", dependencies=[], compute_backward_flow=False, uhd=False)

    frames = np.array(video[0:1])
    right_frames = np.array(video[1:2])
    y_rife = rife_repr(frames, right_frames)
    assert y_rife.shape == (1, 32, 64, 2), y_rife.shape

    y_rife_images = rife_repr.make_images(frames, y_rife)
    assert y_rife_images.shape == (1, 64, 128, 3), y_rife_images.shape
    assert y_rife_images.dtype == np.uint8, y_rife_images.dtype

def test_rife_uhd_true():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    rife_repr = FlowRife(name="rife", dependencies=[], compute_backward_flow=False, uhd=True)

    frames = np.array(video[0:1])
    right_frames = np.array(video[1:2])
    y_rife = rife_repr(frames, right_frames)
    assert y_rife.shape == (1, 16, 32, 2), y_rife.shape

    y_rife_images = rife_repr.make_images(frames, y_rife)
    assert y_rife_images.shape == (1, 64, 128, 3), y_rife_images.shape
    assert y_rife_images.dtype == np.uint8, y_rife_images.dtype

if __name__ == "__main__":
    test_rife_uhd_false()
