"""frame_writer module. Other readers will implement these methods."""
from abc import ABC, abstractmethod
from pathlib import Path

class FrameWriter(ABC):
    """FrameWriter - defines the interface of a reader"""
    @abstractmethod
    def write(self, video: "VREVideo", out_path: str | Path, start_frame: int = 0, end_frame: int | None = None):
        """writes video frames to the disk"""
