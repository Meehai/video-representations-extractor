from tempfile import TemporaryDirectory
import numpy as np
import pytest
from vre_video.utils import image_write
from vre_video.readers import build_frame_reader, PILFrameReader, NumpyFrameReader

@pytest.fixture
def tmpdir(request: pytest.FixtureRequest):
    tmpdir = TemporaryDirectory()
    request.addfinalizer(lambda: tmpdir.cleanup())
    yield tmpdir.name

def test_build_framereader_PILFrameReader(tmpdir: str):
    for i in range(10):
        image_write(np.zeros((30, 30, 3), dtype=np.uint8), f"{tmpdir}/{i}.png")
    frame_reader = build_frame_reader(tmpdir)
    assert isinstance(frame_reader, PILFrameReader), type(frame_reader)
    # calling it with pre-intantiated object returns the same object
    assert isinstance(f2 := build_frame_reader(frame_reader), PILFrameReader), type(f2)
    assert frame_reader.shape == (10, 30, 30, 3)

def test_build_framereader_NumpyFrameReader_npy(tmpdir: str):
    for i in range(10):
        np.save(f"{tmpdir}/{i}.npy", np.zeros((30, 30, 3), dtype=np.uint8))
    frame_reader = build_frame_reader(tmpdir)
    assert isinstance(frame_reader, NumpyFrameReader), type(frame_reader)
    assert isinstance(f2 := build_frame_reader(frame_reader), NumpyFrameReader), type(f2)
    assert f2.shape == (10, 30, 30, 3)

def test_build_framereader_NumpyFrameReader_npz(tmpdir: str):
    for i in range(10):
        np.savez(f"{tmpdir}/{i}.npz", np.zeros((30, 30, 3), dtype=np.uint8))
    frame_reader = build_frame_reader(tmpdir)
    assert isinstance(frame_reader, NumpyFrameReader), type(frame_reader)
    assert frame_reader.shape == (10, 30, 30, 3)
