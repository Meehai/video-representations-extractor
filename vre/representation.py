"""VRE Representation module"""
from __future__ import annotations
import numpy as np
import pims
import cv2
from pathlib import Path
from abc import ABC, abstractmethod
from functools import lru_cache

RepresentationOutput = np.ndarray | tuple[np.ndarray, dict]

class Representation(ABC):
    """Generic Representation class for VRE"""
    def __init__(self, video: pims.Video, name: str, dependencies: list[Representation]):
        assert isinstance(dependencies, (set, list))
        self.name = name
        self.dependencies = dependencies
        # This attribute is checked in the main vre object to see that this parent constructor was called properly.
        self.video = video

    @abstractmethod
    def make(self, t: slice) -> RepresentationOutput:
        """
        Main method of this representation. Calls the internal representation's logic to transform the current provided
        RGB frame of the attached video into the output representation.
        Note: The representation that is returned is guaranteed to be a float32 in the range of [0-1].
        Some representations may not be natively represented like that (i.e. optical flow being [-1:1]). The end-user
        must take care of this cases.

        The returned value is either a simple numpy array of the same shape as the video plus an optional tuple with
        extra stuff. This extra stuff is whatever that representation may want to store about the frames it was given.

        This is also invoked for repr[t] and repr(t).
        """

    @abstractmethod
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        """Given the output of self.make(t), which a [0:1] float32 numpy array, return a [0:255] uint8 image"""

    def resize(self, x: np.ndarray, height: int, width: int) -> np.ndarray:
        """
        Resizes a representation output made from self.make(t). Info about time may be passed via 'extra' dict.
        Update this for more complex cases.
        """
        y = np.array([cv2.resize(_x, (width, height), interpolation=cv2.INTER_LINEAR) for _x in x])
        return y

    def __getitem__(self, t: slice | int) -> np.ndarray:
        return self.__call__(t)

    # @lru_cache(maxsize=1000)
    def __call__(self, t: slice | int) -> RepresentationOutput:
        if isinstance(t, int):
            t = slice(t, t + 1)
        assert t.start >= 0 and t.stop < len(self.video), t
        # Get the raw result of this representation
        res = self.make(t)
        raw_data, extra = res if isinstance(res, tuple) else (res, {})
        # uint8 for semantic segmentation, float32 for everything else
        assert raw_data.dtype in (np.float32, np.uint8), raw_data.dtype
        if raw_data.dtype == np.float32:
            assert raw_data.min() >= 0 and raw_data.max() <= 1, (f"{self.name}: [{raw_data.min():.2f}:"
                                                                 f"{raw_data.max():.2f}]")
        return raw_data, extra

    # TODO: see if this is needed for more special/faster resizes
    # def resizeRawData(self, rawData: np.ndarray) -> np.ndarray:
    #     interpolation = "nearest" if rawData.dtype == np.uint8 else "bilinear"
    #     # OpenCV bugs with uint8 and nearest, adding 255 values (in range [0-1])
    #     dtype = np.int32 if rawData.dtype == np.uint8 else rawData.dtype
    #     resizedData = image_resize(
    #         rawData.astype(dtype),
    #         height=self.outShape[0],
    #         width=self.outShape[1],
    #         only_uint8=False,
    #         interpolation=interpolation,
    #     )
    #     resizedData = resizedData.astype(rawData.dtype)
    #     return resizedData
