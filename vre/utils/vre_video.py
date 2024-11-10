"""FakeVideo module"""
from typing import Union
from slicerator import Slicerator
import numpy as np
import pims

class FakeVideo(Slicerator):
    """FakeVideo -- class used to test representations with a given numpy array"""
    def __init__(self, data: np.ndarray, frame_rate: int):
        assert len(data) > 0, "No data provided"
        super().__init__(data, range(len(data)), len(data))
        self.data = data
        self.frame_rate = frame_rate
        self.frame_shape = data.shape[1:]
        self.file = f"FakeVideo {self.data.shape}"

    @property
    def shape(self):
        """returns the shape of the data"""
        return self.data.shape

    def __repr__(self):
        return f"[FakeVideo] Len: {len(self.data)}. Frame shape: {self.data.shape[1:]}. Frame rate: {self.frame_rate}."

VREVideo = Union[pims.FramesSequence, FakeVideo]
