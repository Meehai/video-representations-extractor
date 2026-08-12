import numpy as np
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from vre_video.writers.numpy_frame_writer import NumpyFrameWriter

class DummyVideo:
    """Minimal video-like object to simulate VREVideo interface"""
    def __init__(self, frames):
        self.frames = frames
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, ix):
        return self.frames[ix]

@pytest.fixture
def tmpdir(request: pytest.FixtureRequest):
    tmpdir = TemporaryDirectory()
    request.addfinalizer(lambda: tmpdir.cleanup())
    yield Path(tmpdir.name)

@pytest.mark.parametrize(["fmt", "compress"], [("npy", False), ("npz", False), ("npz", True), ("npy", True)])
def test_NumpyFrameWriter_write_basic(tmpdir: Path, fmt: str, compress):
    frames = [np.full((16, 16, 3), i, dtype=np.uint8) for i in range(5)]
    video = DummyVideo(frames)

    if fmt == "npy" and compress is True:
        with pytest.raises(AssertionError):
            _ = NumpyFrameWriter(fmt=fmt, compress=compress)
        return

    writer = NumpyFrameWriter(fmt=fmt, compress=compress)
    writer.write(video, out_dir := tmpdir / "out")

    files = sorted(out_dir.glob("*"))
    assert len(files) == len(frames)
    for file in files:
        assert file.is_file()
        assert file.suffix[1:] == fmt
