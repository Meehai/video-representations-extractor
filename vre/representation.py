"""VRE Representation module"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from .utils import parsed_str_type, RepresentationOutput
from .vre_representation_mixin import VRERepresentationMixin

class Representation(ABC, VRERepresentationMixin):
    """Generic Representation class for VRE"""
    def __init__(self, name: str, dependencies: list[Representation]):
        super().__init__()
        assert isinstance(dependencies, (set, list))
        self.name = name
        self.dependencies = dependencies

    @abstractmethod
    def make(self, frames: np.ndarray, dep_data: dict[str, RepresentationOutput] | None = None) -> RepresentationOutput:
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

    @abstractmethod
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        """
        Resizes the output of a self.make(frames) call into some other resolution
        Parameters:
        - repr_data The original representation output
        - new_size A tuple of two positive integers representing the new size
        Returns: A new representation output at the desired size
        """

    @abstractmethod
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        """Returns the (h, w) tuple of the size of the current representation"""

    def __getitem__(self, t: slice | int) -> RepresentationOutput:
        return self.__call__(self.video[t])

    def __call__(self, *args, **kwargs) -> RepresentationOutput:
        return self.make(*args, **kwargs)

    def __repr__(self):
        return f"[Representation] {parsed_str_type(self)}({self.name})"
