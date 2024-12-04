from vre.utils.vre_video import FakeVideo
import numpy as np
import pytest

def test_FakeVideo_ctor():
    fake_video = FakeVideo(data := np.random.randint(0, 256, size=(10, 20, 30, 3), dtype=np.uint8), fps=1)
    assert len(fake_video) == 10 and fake_video.fps == 1 and fake_video.frame_shape == (20, 30, 3)
    assert np.allclose(data[2], fake_video[2])

def test_FakeVideo_frames():
    fake_video = FakeVideo(data := np.random.randint(0, 256, size=(10, 20, 30, 3), dtype=np.uint8), fps=1,
                           frames=[2, 5, *list(range(50, 58))])
    assert len(fake_video) == 58 and fake_video.fps == 1 and fake_video.frame_shape == (20, 30, 3)
    with pytest.raises(KeyError):
        _ = fake_video[49]
    with pytest.raises(KeyError):
        _ = fake_video[0]
    assert np.allclose(data[0], fake_video[2])
    assert np.allclose(data[1], fake_video[5])
    assert np.allclose(data[2], fake_video[50])
