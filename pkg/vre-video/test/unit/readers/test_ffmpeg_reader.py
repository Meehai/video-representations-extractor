from pathlib import Path
import numpy as np
import pytest
from vre_video.utils import fetch
from vre_video.readers import FFmpegFrameReader

@pytest.fixture
def video_path() -> str:
    url = "https://gitlab.com/video-representations-extractor/video-representations-extractor/-/raw/master/resources/test_video.mp4"
    if not (dst := Path(__file__).parents[2] / "test_video.mp4").exists():
        fetch(url, dst)
    return str(dst)

def _fake_reader(monkeypatch, tmp_path, n_frames: int, actual_frames: int, cache_len: int, fps: float = 30.0):
    """Build an FFmpegFrameReader whose ffmpeg process is faked: the pipe outputs 'actual_frames'
    frames, one per raw frame, with frame i having content bytes([i, i, i]). 'n_frames' is what
    ffprobe reports (len(self)) and may differ from 'actual_frames'."""
    class FakeFFprobe:
        def __init__(self, _path):
            self.fps = fps
            self.shape = (n_frames, 1, 1, 3)

    class FakeFrames: # a rawvideo pipe: frame i has content bytes([i, i, i])
        def __init__(self, start_frame: int):
            self._next = start_frame
        def read(self, _n_bytes: int) -> bytes:
            if self._next >= actual_frames:
                return b""
            frame = self._next
            self._next += 1
            return bytes([frame] * 3)
        def close(self):
            pass

    class FakeProcess:
        def __init__(self, start_time: float):
            self.stdout = FakeFrames(int(start_time * fps))
        def terminate(self):
            pass

    monkeypatch.setattr("vre_video.readers.ffmpeg_frame_reader.FFprobe", FakeFFprobe)
    monkeypatch.setattr(FFmpegFrameReader, "_start_ffmpeg_process",
                        lambda self, start_time: setattr(self, "process", FakeProcess(start_time)))

    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    return FFmpegFrameReader(str(video), cache_len=cache_len)

def test_FFmpegFrameReader_ctor(video_path: str):
    reader = FFmpegFrameReader(video_path)
    assert reader.path == video_path
    assert reader.shape == (5395, 720, 1280, 3)
    assert reader.fps == 29.97002997002997
    assert reader[0].shape == (720, 1280, 3)

def test_FFmpegFrameReader_sequential_read_all_frames(tmp_path, monkeypatch):
    """Regression test for the Bug #6 off-by-one (see .tracker/todos/open/6): the _cache_frames
    read loop consumed one extra frame per cache window, drifting the cache bookkeeping one frame
    behind the real stream per window until a continuation read hit EOF and crashed with
    'list index out of range'. Fixed; the marker was removed. Pure unit test - no ffmpeg,
    the process is faked."""
    n_frames = 30
    reader = _fake_reader(monkeypatch, tmp_path, n_frames, n_frames, cache_len=3)

    for ix in range(n_frames):
        assert np.array_equal(reader[ix], np.full((1, 1, 3), ix, dtype=np.uint8)), \
            f"frame {ix}: got a shifted or skipped frame"

def test_FFmpegFrameReader_stream_shorter_than_reported(tmp_path, monkeypatch):
    """Regression test for the Bug #6 EOF trap (see .tracker/todos/open/6): when len(self)
    overestimates the actual output frames (VFR / metadata mismatch / decode drops), reading past
    the real end of the stream must raise a clear error, not leave an empty cache with
    cache_start == cache_end == ix that IndexErrors on the next access."""
    n_frames, actual_frames = 30, 27
    reader = _fake_reader(monkeypatch, tmp_path, n_frames, actual_frames, cache_len=3)

    with pytest.raises(EOFError, match="ffprobe"):
        for ix in range(n_frames):
            reader[ix]
