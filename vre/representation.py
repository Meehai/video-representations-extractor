"""VRE Representation module"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pims
# from functools import lru_cache

from .utils import parsed_str_type

RepresentationOutput = np.ndarray | tuple[np.ndarray, list[dict]]

class Representation(ABC):
    """Generic Representation class for VRE"""
    def __init__(self, video: pims.Video, name: str, dependencies: list[Representation]):
        assert isinstance(dependencies, (set, list))
        self.name = name
        self.dependencies = dependencies
        # This attribute is checked in the main vre object to see that this parent constructor was called properly.
        self.video = video

    @abstractmethod
    def vre_setup(self, **kwargs):
        """
        Setup method for this representation. This is required to run this representation from within VRE.
        We do this setup separately, so we can instatiate the representation without doing any VRE specific setup,
        like loading weights.
        """

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
    def make_images(self, t: slice, x: np.ndarray, extra: dict | None) -> np.ndarray:
        """Given the output of self.make(t), which a [0:1] float32 numpy array, return a [0:255] uint8 image"""

    def __getitem__(self, t: slice | int) -> RepresentationOutput:
        return self.__call__(t)

    # @lru_cache(maxsize=1000)
    def __call__(self, t: slice | int) -> RepresentationOutput:
        if isinstance(t, int):
            t = slice(t, t + 1)
        assert t.start >= 0 and t.stop <= len(self.video), f"Video len: {len(self.video)}, slice: {t}"
        # Get the raw result of this representation
        res = self.make(t)
        raw_data, extra = res if isinstance(res, tuple) else (res, {})
        # uint8 for semantic segmentation, float32 for everything else
        assert raw_data.dtype in (np.float32, np.uint8), raw_data.dtype
        # if raw_data.dtype == np.float32:
        #     assert raw_data.min() >= 0 and raw_data.max() <= 1, (f"{self.name}: [{raw_data.min():.2f}:"
        #                                                          f"{raw_data.max():.2f}]")
        return raw_data, extra

    def __repr__(self):
        return f"[{parsed_str_type(self)} VRE Representation: {self.name}]"
