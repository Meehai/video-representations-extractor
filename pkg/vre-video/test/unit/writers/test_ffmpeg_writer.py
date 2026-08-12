import numpy as np
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from vre_video.readers.ffmpeg_frame_reader import FFmpegFrameReader
from vre_video.writers.ffmpeg_frame_writer import FFmpegFrameWriter

class DummyVideo:
    """Minimal video-like object to simulate VREVideo interface"""
    def __init__(self, frames):
        self.frames: list[np.ndarray] = frames
        self.shape = (len(self.frames), *self.frames[0].shape)
        self.fps = 1
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, ix):
        return self.frames[ix]

@pytest.fixture
def tmpdir(request: pytest.FixtureRequest):
    tmpdir = TemporaryDirectory()
    request.addfinalizer(lambda: tmpdir.cleanup())
    yield Path(tmpdir.name)

def test_FFmpegFrameWriter_write_basic(tmpdir: Path):
    frames = [np.full((16, 16, 3), i, dtype=np.uint8) for i in range(5)]
    video = DummyVideo(frames)

    writer = FFmpegFrameWriter()
    writer.write(video, out_file := tmpdir / "out.mp4")
    assert out_file.exists()

    video2 = FFmpegFrameReader(out_file)
    assert video.shape == video2.shape
