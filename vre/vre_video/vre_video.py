"""VREVideo module"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable
import numpy as np

class VREVideo(Iterable, ABC):
    """VREVideo -- A generic wrapper on top of a Video container"""
    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """the shape of a frame"""
        return self.shape[1:]

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int, int]:
        """Returns the (N, H, W, 3) tuple of the video"""

    @property
    @abstractmethod
    def fps(self) -> float:
        """The frame rate of the video"""

    @abstractmethod
    def get_one_frame(self, ix: int) -> np.ndarray:
        """Gets one frame of this video"""

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

    def __getitem__(self, ix: int | list[int] | np.ndarray | slice) -> np.ndarray:
        if isinstance(ix, np.ndarray):
            return self[ix.tolist()]
        if isinstance(ix, list):
            return np.array([self[_ix] for _ix in ix])
        if isinstance(ix, slice):
            return np.array([self[_ix] for _ix in range(ix.start, ix.stop)])
        return self.get_one_frame(ix)
