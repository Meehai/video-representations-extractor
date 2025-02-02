"""VREVideo module"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable
from tqdm import trange
import ffmpeg

class VREVideo(Iterable, ABC):
    """VREVideo -- A generic wrapper on top of a Video container"""
    def __init__(self):
        self.write_process = None

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int, int]:
        """Returns the (N, H, W, 3) tuple of the video"""

    @property
    @abstractmethod
    def fps(self) -> float:
        """The frame rate of the video"""

    def write(self, out_path: Path, start_frame: int = 0, end_frame: int | None = None):
        """writes the video to the path"""
        out_path = Path(out_path)
        assert self.write_process is None, self.write_process
        assert out_path.suffix == ".mp4", out_path
        assert isinstance(start_frame, int) and start_frame >= 0, start_frame

        self.write_process = (
            ffmpeg
            .input("pipe:0", format="rawvideo", pix_fmt="rgb24", s=f"{self.shape[2]}x{self.shape[1]}", r=self.fps)
            .output(str(out_path), pix_fmt="yuv420p", vcodec="libx264")
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=-3, pipe_stdout=-3) # -3 = subprocess.DEVNULL
        )

        try:
            for frame_ix in trange(start_frame, end_frame or len(self)):
                self.write_process.stdin.write(self[frame_ix].tobytes())
        finally:
            self.write_process.stdin.close()
            self.write_process.wait()
            self.write_process = None

    def __iter__(self):
        index = 0
        try:
            while True:
                yield self[index]
                index += 1
        except IndexError:
            pass
