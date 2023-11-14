"""VRE Representation module"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from .vre_representation_mixin import VRERepresentationMixin
from .utils import parsed_str_type

RepresentationOutput = np.ndarray | tuple[np.ndarray, list[dict]]

class Representation(VRERepresentationMixin, ABC):
    """Generic Representation class for VRE"""
    def __init__(self, name: str, dependencies: list[Representation]):
        assert isinstance(dependencies, (set, list))
        self.name = name
        self.dependencies = dependencies

    @abstractmethod
    def make(self, frames: np.ndarray) -> RepresentationOutput:
        """
        Main method of this representation. Calls the internal representation's logic to transform the current provided
        RGB frame of the attached video into the output representation.
        Note: The representation that is returned is guaranteed to be a float32 (or uint8) numpy array.

        The returned value is either a simple numpy array of the same shape as the video plus an optional tuple with
        extra stuff. This extra stuff is whatever that representation may want to store about the frames it was given.

        This is also invoked for repr[t] and repr(t).
        """

    @abstractmethod
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        """Given the output of self.make(frames) of type RepresentationOutput, return a [0:255] image for each frame"""

    def __getitem__(self, t: slice | int) -> RepresentationOutput:
        return self.__call__(t)

    def __call__(self, *args, **kwargs) -> RepresentationOutput:
        return self.make(*args, **kwargs)

    def __repr__(self):
        return f"[{parsed_str_type(self)} VRE Representation: {self.name}]"
