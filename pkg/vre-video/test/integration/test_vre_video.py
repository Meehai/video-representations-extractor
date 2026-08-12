from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pytest
from vre_video import VREVideo
from vre_video.utils import image_write, fetch
from vre_video.readers import PILFrameReader, NumpyFrameReader, FFmpegFrameReader
from vre_video.writers import PILFrameWriter, NumpyFrameWriter, FFmpegFrameWriter

@pytest.fixture
def pngdir(request: pytest.FixtureRequest):
    tmpdir = TemporaryDirectory()
    request.addfinalizer(lambda: tmpdir.cleanup())
    for i in range(9):
        image_write(np.zeros((30, 30, 3), dtype=np.uint8), f"{tmpdir.name}/{i}.png")
    yield tmpdir.name

@pytest.fixture
def npydir(request: pytest.FixtureRequest):
    tmpdir = TemporaryDirectory()
    request.addfinalizer(lambda: tmpdir.cleanup())
    for i in range(9):
        np.save(f"{tmpdir.name}/{i}.png", np.zeros((30, 30, 3), dtype=np.uint8))
    yield tmpdir.name

@pytest.fixture
def video_path() -> str:
    url = "https://gitlab.com/video-representations-extractor/video-representations-extractor/-/raw/master/resources/test_video.mp4"
    if not (dst := Path(__file__).parents[2] / "test_video.mp4").exists():
        fetch(url, dst)
    return dst

def test_VREVideo_PIL(pngdir: str):
    video = VREVideo(pngdir)
    assert isinstance(video.reader, PILFrameReader), type(video.reader)
    assert video.shape == (9, 30, 30, 3)
    assert video.frame_shape == (30, 30, 3)
    assert video.fps == 1

    video = VREVideo(pngdir, fps=3)
    assert video.fps == 3

    writer = video.write(out_path := f"{pngdir}/write", writer="PIL")
    assert isinstance(writer, PILFrameWriter)
    files = sorted(Path(out_path).iterdir(), key=lambda p: p.name)
    assert len(files) == 9
    for i, file in enumerate(files):
        assert file.name == f"{i}.png"

def test_VREVideo_Numpy(npydir: str):
    video = VREVideo(npydir)
    assert isinstance(video.reader, NumpyFrameReader), type(video.reader)
    assert video.shape == (9, 30, 30, 3)
    assert video.frame_shape == (30, 30, 3)
    assert video.fps == 1

    res = video.write(out_path := f"{npydir}/write", writer="np", fmt="npy")
    assert isinstance(res, NumpyFrameWriter), type(res)
    assert res.format == "npy"
    files = sorted(Path(out_path).iterdir(), key=lambda p: p.name)
    assert len(files) == 9
    for i, file in enumerate(files):
        assert file.name == f"{i}.npy"

def test_VREVideo_FFmpeg(video_path: str):
    video = VREVideo(video_path)
    assert isinstance(video.reader, FFmpegFrameReader), type(video.reader)
    assert video.shape == (5395, 720, 1280, 3)
    assert video.fps == 29.97002997002997

    with TemporaryDirectory() as tmp:
        res = video.write(out_path := f"{tmp}/write.mp4", start_frame=100, end_frame=120)
        assert isinstance(res, FFmpegFrameWriter), type(res)

        video2 = VREVideo(out_path)
        assert video.frame_shape == video2.frame_shape
        assert len(video2) == 20
