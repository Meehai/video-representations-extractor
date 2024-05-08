import numpy as np
from vre.representations.optical_flow.rife import FlowRife
from vre.utils import FakeVideo

def test_rife_uhd_false():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    uhd = bool(np.random.randint(0, 2))
    rife_repr = FlowRife(name="rife", dependencies=[], compute_backward_flow=False, uhd=uhd)

    frames = np.array(video[0:1])
    right_frames = np.array(video[1:2])
    y_rife = rife_repr(frames, right_frames)
    out_shape = (32, 64) if not uhd else (16, 32)
    assert y_rife.shape == (1, *out_shape, 2), y_rife.shape

    y_rife_images = rife_repr.make_images(frames, y_rife)
    assert y_rife_images.shape == (1, *out_shape, 3), y_rife_images.shape
    assert y_rife_images.dtype == np.uint8, y_rife_images.dtype

    assert rife_repr.size(y_rife) == out_shape
    y_rife_resized = rife_repr.resize(y_rife, (64, 128)) # we can resize it though
    assert rife_repr.size(y_rife_resized) == (64, 128)
    assert rife_repr.make_images(frames, y_rife_resized).shape == (1, 64, 128, 3)
