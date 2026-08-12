import numpy as np
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from vre_video.writers.pil_frame_writer import PILFrameWriter

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

@pytest.mark.parametrize("fmt", ["png", "jpg"])
def test_pil_framewriter_writes_images(tmpdir: Path, fmt: str):
    frames = [np.full((16, 16, 3), i, dtype=np.uint8) for i in range(5)]
    video = DummyVideo(frames)

    writer = PILFrameWriter(fmt=fmt)
    writer.write(video, out_dir := tmpdir / "out")

    files = sorted(out_dir.glob("*"))
    assert len(files) == len(frames)
    for file in files:
        assert file.is_file()
        assert file.suffix[1:] == fmt

def test_pil_framewriter_raises_on_nonempty_output(tmpdir: Path):
    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    video = DummyVideo(frames)

    out_dir = tmpdir / "out"
    out_dir.mkdir()
    (out_dir / "dummy.png").touch()  # simulate existing file

    writer = PILFrameWriter()
    with pytest.raises(AssertionError, match="Data exists in out_path"):
        writer.write(video, out_dir)

def test_pil_framewriter_writes_images_interval(tmpdir: Path):
    frames = [np.full((16, 16, 3), i, dtype=np.uint8) for i in range(5)]
    video = DummyVideo(frames)

    writer = PILFrameWriter(fmt="png")
    with pytest.raises(AssertionError):
        writer.write(video, out_dir := tmpdir / "out", start_frame=-1)
    with pytest.raises(AssertionError):
        writer.write(video, out_dir := tmpdir / "out", end_frame=6)
    writer.write(video, out_dir := tmpdir / "out", start_frame=2, end_frame=4)

    files = sorted(out_dir.glob("*"))
    assert [f.name for f in files] == ["2.png", "3.png"], files
