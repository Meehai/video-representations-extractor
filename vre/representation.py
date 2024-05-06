"""VRE Representation module"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from .utils import parsed_str_type, VREVideo
from .logger import logger

RepresentationOutput = np.ndarray | tuple[np.ndarray, list[dict]]

class Representation(ABC):
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

    ## Optional methods ##

    # pylint: disable=unused-argument
    def vre_setup(self, video: VREVideo, **kwargs):
        """
        Setup method for this representation. This is required to run this representation from within VRE.
        We do this setup separately, so we can instatiate the representation without doing any VRE specific setup,
        like loading weights.
        """
        logger.debug(f"[{parsed_str_type(self)} No runtime setup provided.")

    # pylint: disable=unused-argument
    def vre_dep_data(self, video: VREVideo, ix: slice) -> dict[str, RepresentationOutput]:
        """method used to retrieve the dependencies' data for this frames during a vre run"""
        return {}

    def __getitem__(self, t: slice | int) -> RepresentationOutput:
        return self.__call__(t)

    def __call__(self, *args, **kwargs) -> RepresentationOutput:
        return self.make(*args, **kwargs)

    def __repr__(self):
        return f"[VRE Representation] {parsed_str_type(self)}({self.name})"
