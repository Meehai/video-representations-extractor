"""VRE Representation module"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from ..utils import parsed_str_type, lo

MemoryData = np.ndarray

@dataclass
class ReprOut:
    """The output of representation.compute()"""
    frames: np.ndarray | None
    output: MemoryData
    key: list[int]
    extra: list[dict] | None = None
    output_images: np.ndarray | None = None

    def __post_init__(self):
        assert isinstance(self.output, np.ndarray), type(self.output)
        assert len(self.output) == len(self.key), (len(self.output), len(self.key))
        assert self.frames is None or len(self.frames) == len(self.output), (len(self.frames), len(self.output))

    def __repr__(self):
        return (f"[ReprOut](key={self.key}, output={lo(self.output)}, output_images={lo(self.output_images)} "
                f"extra set={self.extra is not None}, frames={lo(self.frames)})")

class Representation(ABC):
    """Generic Representation class for VRE"""
    def __init__(self, name: str, dependencies: list[Representation] | None = None):
        deps = [] if dependencies is None else dependencies
        assert isinstance(deps, list), type(deps)
        assert all(isinstance(dep, Representation) for dep in deps), (name, [(dep, type(dep)) for dep in deps])
        self.name = name
        self.dependencies = deps
        self.data: ReprOut | None = None

    ## Abstract methods ##
    @abstractmethod
    def make_images(self) -> np.ndarray:
        """Given the output of self.compute(video, ixs) of type ReprOut, return a [0:255] image for each frame"""

    ## Public methods & properties ##
    @property
    def dep_names(self) -> list[str]:
        """return the list of dependencies names"""
        return [r.name for r in self.dependencies]

    @property
    def size(self) -> tuple[int, ...]:
        """Returns the (b, h, w, c) tuple of the size of the current representation"""
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        return tuple(self.data.output.shape)

    @property
    def is_classification(self) -> bool:
        """if we have self.classes. Used in MultiTaskReader."""
        return hasattr(self, "classes") and self.classes is not None # pylint: disable=no-member

    ## Magic methods ##
    def __repr__(self):
        return f"{parsed_str_type(self)}({self.name}{f' {self.dep_names}' if len(self.dep_names) > 0 else ''})"
