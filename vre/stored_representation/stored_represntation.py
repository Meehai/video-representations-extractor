"""StoredRepresentations: representations that are files on the disk"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch as tr

from vre.representations import ReprOut

class StoredRepresentation(ABC):
    """StoredRepresentation. The counterpart to Representation which is ComputedRepresentation. TBD how to integrate"""
    def __init__(self, name: str, n_channels: int, dependencies: list[StoredRepresentation] | None = None):
        self.name = name
        self.n_channels = n_channels
        dependencies = deps = [self] if dependencies is None else dependencies
        assert all(isinstance(dep, StoredRepresentation) for dep in deps), f"{self}: {dict(zip(deps, map(type, deps)))}"
        self.dependencies: list[StoredRepresentation] = dependencies
        self.classes: list[str] | None = None
        self.data: ReprOut | None = None # TODO: dummy

    @abstractmethod
    def return_fn(self, load_data: tr.Tensor) -> tr.Tensor:
        """return_fn is the code that's ran between what's stored on the disk and what's actually sent to the model"""

    @abstractmethod
    def load_from_disk(self, path: Path) -> tr.Tensor:
        """Reads the data from the disk and transforms it properly"""

    @abstractmethod
    def save_to_disk(self, data: tr.Tensor, path: Path):
        """Stores the data to disk"""

    @abstractmethod
    def plot_fn(self, x: tr.Tensor) -> np.ndarray:
        """plots this representation. TBD: merge with Representation.make_images()"""

    @property
    def is_classification(self) -> bool:
        """if we have self.classes"""
        return self.classes is not None

    @property
    def dep_names(self) -> list[str]:
        """The names of the dependencies of this representation"""
        return [dep.name for dep in self.dependencies]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{str(type(self)).split('.')[-1][0:-2]}({self.name}[{self.n_channels}])"
