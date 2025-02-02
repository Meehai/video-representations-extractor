"""VREVideo module"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

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

    @abstractmethod
    def write(self, out_path: Path, start_frame: int = 0, end_frame: int | None = None):
        """writes the video to the path. TODO: add params to send to the underlying write impl (ffmpeg/image_write)"""

    def __iter__(self):
        index = 0
        try:
            while True:
                yield self[index]
                index += 1
        except IndexError: # why?
            pass
