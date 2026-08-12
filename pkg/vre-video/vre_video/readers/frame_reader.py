"""frame_reader module. Other readers will implement these methods."""
from abc import ABC, abstractmethod
import numpy as np
from vre_video.utils import parsed_str_type

class FrameReader(ABC):
    """FrameReader - defines the interface of a reader"""
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int, int]:
        """Returns the (N, H, W, 3) tuple of the video"""

    @property
    @abstractmethod
    def fps(self) -> float:
        """The frame rate of the video"""

    @property
    @abstractmethod
    def path(self) -> str:
        """The path of the video"""

    @property
    def frame_shape(self):
        """The frame shape"""
        return self.shape[1:]

    @abstractmethod
    def __getitem__(self, ix: int | list[int] | np.ndarray | slice) -> np.ndarray:
        """gets one or more frames"""

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return (f"Reader: {parsed_str_type(self)}. Path: {self.path}. FPS: {self.fps}. "
                f"Len: {len(self)}. Frame shape: {self.frame_shape}.")
