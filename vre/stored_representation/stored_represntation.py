"""MultiTask representations stored as .npz files on the disk"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch as tr
from overrides import overrides
from loggez import loggez_logger as logger

from ..utils import FixedSizeOrderedDict

_CACHE = FixedSizeOrderedDict(maxlen=1024)

class StoredRepresentation(ABC):
    """StoredRepresentation. The counterpart to Representation which is ComputedRepresentation. TBD how to integrate"""
    def __init__(self, name: str, n_channels: int, dependencies: list[StoredRepresentation] | None = None):
        self.name = name
        self.n_channels = n_channels
        dependencies = deps = [self] if dependencies is None else dependencies
        assert all(isinstance(dep, StoredRepresentation) for dep in deps), f"{self}: {dict(zip(deps, map(type, deps)))}"
        self.dependencies: list[StoredRepresentation] = dependencies
        self.classes: list[str] | None = None

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

class NpzRepresentation(StoredRepresentation):
    """Generic Task with data read from/saved to npz files. Tries to read data as-is from disk and store it as well"""
    def __init__(self, name: str, n_channels: int, dependencies: list[NpzRepresentation] | None = None):
        super().__init__(name, n_channels, dependencies)

    @overrides
    def load_from_disk(self, path: Path) -> tr.Tensor:
        """Reads the npz data from the disk and transforms it properly"""
        if (key := (self.name, path.name)) in _CACHE:
            logger.debug2(f"HIT: '{key}'")
            return _CACHE[key]
        logger.debug2(f"MISS: '{key}'")
        data = np.load(path, allow_pickle=False)
        data = data if isinstance(data, np.ndarray) else data["arr_0"] # in case on npz, we need this as well
        data = data.astype(np.float32) if np.issubdtype(data.dtype, np.floating) else data # float16 is dangerous
        res = tr.from_numpy(data)
        res = res.unsqueeze(-1) if len(res.shape) == 2 and self.n_channels == 1 else res # (H, W) in some dph/edges
        assert ((res.shape[-1] == self.n_channels and len(res.shape) == 3) or
                (len(res.shape) == 2 and self.is_classification)), f"{self.name}: {res.shape} vs {self.n_channels}"
        _CACHE[key] = res
        return res

    @overrides
    def save_to_disk(self, data: tr.Tensor, path: Path):
        """stores this item to the disk which can then be loaded via `load_from_disk`"""
        np.save(path, data.cpu().detach().numpy(), allow_pickle=False)
