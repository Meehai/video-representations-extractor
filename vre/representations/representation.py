"""VRE Representation module"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from vre_video import VREVideo

from vre.utils import parsed_str_type, ReprOut, DiskData, MemoryData
from vre.logger import vre_logger as logger
from .io_representation import IORepresentation

class Representation(IORepresentation, ABC):
    """Generic Representation class for VRE"""
    def __init__(self, name: str, dependencies: list[Representation] | None = None):
        IORepresentation.__init__(self)
        deps = [] if dependencies is None else dependencies
        assert isinstance(deps, list), type(deps)
        self.name = name
        self.dependencies = deps
        self._batch_size: int | None = None

    @abstractmethod
    def make_images(self, data: ReprOut) -> np.ndarray:
        """Given the output of self.compute(video, ixs) of type ReprOut, return a [0:255] image for each frame"""

    @abstractmethod
    def resize(self, data: ReprOut, new_size: tuple[int, int]):
        """resizes the data. size is provided in (h, w)"""

    @abstractmethod
    def load_from_disk(self, path: Path) -> DiskData:
        """Reads the data from the disk into disk_fmt"""

    @abstractmethod
    def save_to_disk(self, memory_data: MemoryData, path: Path):
        """Stores the disk_fmt data to disk"""

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """The number of channels (dimensions, not resolution) of this representation"""

    ## Public properties & methods

    @property
    def batch_size(self) -> int:
        """the batch size that is used during the VRE run for computation"""
        if self._batch_size is None:
            logger.debug(f"[{self}] No batch_size set, returning 1. Call set_compute_params")
            return 1
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs: int):
        assert isinstance(bs, int) and bs >= 1, type(bs)
        self._batch_size = bs

    @property
    def dep_names(self) -> list[str]:
        """return the list of dependencies names"""
        return [r.name for r in self.dependencies]

    @property
    def is_classification(self) -> bool:
        """if we have self.classes. Used in MultiTaskReader."""
        return hasattr(self, "classes") and self.classes is not None # pylint: disable=no-member

    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        """Transforms the data from disk_fmt into memory_fmt (usable in VRE)"""
        return MemoryData(disk_data)

    def memory_to_disk_fmt(self, memory_data: MemoryData) -> DiskData:
        """Transformes the data from memory_fmt (usable in VRE) to disk_fmt"""
        return np.array(memory_data)

    def size(self, repr_out: ReprOut) -> tuple[int, ...]:
        """Returns the (b, h, w, c) tuple of the size of the current representation"""
        return tuple(repr_out.output.shape)

    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        """
        Given input data (batch of images/frewes), compute the output data of this representation.
        Not abstract so we can instantiate testing representations and enable syntax highligting w/o CompputeReprMixin.
        """
        raise NotImplementedError(f"{self} must implement .compute()")

    def set_compute_params(self, **kwargs):
        """set the compute parameters for the representation"""
        attributes = ["batch_size"]
        res = ""
        assert set(kwargs).issubset(attributes), (list(kwargs), attributes)
        for attr in attributes:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
                res += f"\n-{attr}: {kwargs[attr]}"
        if len(res) > 0:
            logger.debug(f"[{self}] Set node specific 'Compute' params:{res}")

    ## Magic methods ##
    def __repr__(self):
        return f"{parsed_str_type(self)}({self.name}{f' {self.dep_names}' if len(self.dep_names) > 0 else ''})"
