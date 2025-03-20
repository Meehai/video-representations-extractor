"""VRE Representation module"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from ..utils import parsed_str_type, ReprOut

class Representation(ABC):
    """Generic Representation class for VRE"""
    def __init__(self, name: str, dependencies: list[Representation] | None = None):
        deps = [] if dependencies is None else dependencies
        assert isinstance(deps, list), type(deps)
        # assert all(isinstance(dep, Representation) for dep in deps), (name, [(dep, type(dep)) for dep in deps])
        self.name = name
        self.dependencies = deps
        # self.data: ReprOut | None = None

    def size(self, repr_out: ReprOut) -> tuple[int, ...]:
        """Returns the (b, h, w, c) tuple of the size of the current representation"""
        return tuple(repr_out.output.shape)

    ## Abstract methods ##
    @abstractmethod
    def make_images(self, data: ReprOut) -> np.ndarray:
        """Given the output of self.compute(video, ixs) of type ReprOut, return a [0:255] image for each frame"""

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """The number of channels (dimensions, not resolution) of this representation"""

    ## Public methods & properties ##
    @property
    def dep_names(self) -> list[str]:
        """return the list of dependencies names"""
        return [r.name for r in self.dependencies]

    @property
    def is_classification(self) -> bool:
        """if we have self.classes. Used in MultiTaskReader."""
        return hasattr(self, "classes") and self.classes is not None # pylint: disable=no-member

    ## Magic methods ##
    def __repr__(self):
        return f"{parsed_str_type(self)}({self.name}{f' {self.dep_names}' if len(self.dep_names) > 0 else ''})"
