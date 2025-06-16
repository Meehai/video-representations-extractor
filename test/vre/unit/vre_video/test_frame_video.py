from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import numpy as np
import pytest
from vre.vre_video import FrameVideo
from vre.utils import image_write

def test_FrameVideo_ctor():
    fake_video = FrameVideo(data := np.random.randint(0, 256, size=(10, 20, 30, 3), dtype=np.uint8), fps=1)
    assert len(fake_video) == 10 and fake_video.fps == 1 and fake_video.frame_shape == (20, 30, 3)
    assert np.allclose(data[2], fake_video[2])

def test_FrameVideo_frames():
    fake_video = FrameVideo(data := np.random.randint(0, 256, size=(10, 20, 30, 3), dtype=np.uint8), fps=1,
                           frames=[2, 5, *list(range(50, 58))])
    assert len(fake_video) == 58 and fake_video.fps == 1 and fake_video.frame_shape == (20, 30, 3)
    with pytest.raises(KeyError):
        _ = fake_video[49]
    with pytest.raises(KeyError):
        _ = fake_video[0]
    assert np.allclose(data[0], fake_video[2])
    assert np.allclose(data[1], fake_video[5])
    assert np.allclose(data[2], fake_video[50])

def test_FrameVideo_write():
    fake_video = FrameVideo(np.random.randint(0, 256, size=(10, 20, 30, 3), dtype=np.uint8), fps=1)
    tempfile = NamedTemporaryFile("w", suffix=".mp4").name
    fake_video.write(tempfile)
    assert Path(tempfile).exists()

def test_FrameVideo_framesdir_npz():
    with TemporaryDirectory() as tmpdir:
        for i in range(10):
            np.savez(f"{tmpdir}/{i}.npz", np.zeros((30, 30, 3), dtype=np.uint8))
        video = FrameVideo(tmpdir, fps=1)
        assert video.shape == (10, 30, 30, 3)

def test_FrameVideo_framesdir_png():
    with TemporaryDirectory() as tmpdir:
        for i in range(10):
            image_write(np.zeros((30, 30, 3), dtype=np.uint8), f"{tmpdir}/{i}.png")
        video = FrameVideo(tmpdir, fps=1)
        assert video.shape == (10, 30, 30, 3)

def test_FrameVideo_framesdir_unsupported():
    with TemporaryDirectory() as tmpdir:
        for i in range(10):
            image_write(np.zeros((30, 30, 3), dtype=np.uint8), f"{tmpdir}/{i}.jpg")
        with pytest.raises(AssertionError):
            _ = FrameVideo(tmpdir, fps=1)

    with TemporaryDirectory() as tmpdir:
        for i in range(10):
            np.save(f"{tmpdir}/{i}.npy", np.zeros((30, 30, 3), dtype=np.uint8))
        with pytest.raises(AssertionError):
            _ = FrameVideo(tmpdir, fps=1)

    # frames cannot be infered
    with TemporaryDirectory() as tmpdir:
        for i in range(10):
            np.savez(f"{tmpdir}/lala-{i}.npz", np.zeros((30, 30, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            _ = FrameVideo(tmpdir, fps=1)

def test_FrameVideo_framesdir_npz_frames():
    with TemporaryDirectory() as tmpdir:
        for i in range(10):
            np.savez(f"{tmpdir}/lala-{i}.npz", np.zeros((30, 30, 3), dtype=np.uint8))
        video = FrameVideo(Path(tmpdir), fps=1, frames=list(range(0, 20, 2))) # explicitly provide frame number (sorted)
        assert video.shape == (10, 30, 30, 3)
        assert video[0].shape == (30, 30, 3)
        with pytest.raises(KeyError):
            _ = video[1]
        assert video[18].shape == (30, 30, 3)
