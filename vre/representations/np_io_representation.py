"""NpzRepresentation -- IO Representation for files stored as npy/npz binary format"""
from __future__ import annotations
from copy import deepcopy
from typing import Callable
from pathlib import Path
from overrides import overrides
import numpy as np
from vre.utils import FixedSizeOrderedDict
from vre.logger import vre_logger as logger

from .io_representation_mixin import IORepresentationMixin, MemoryData, DiskData

_CACHE = FixedSizeOrderedDict(maxlen=1024)

class NpIORepresentation(IORepresentationMixin):
    """Generic Task with data read from/saved to numpy files. Tries to read data as-is from disk and store it as well"""
    @property
    def save_fn(self) -> Callable | None:
        """the function that is used to save the representation based on binary_format and compress"""
        if self.binary_format.value == "npz":
            return np.savez_compressed if self.compress else np.savez
        if self.binary_format.value == "npy":
            assert self.compress is False, "Compress cannot be used with regular np.save() (npy format)"
            return np.save
        return None

    @overrides
    def load_from_disk(self, path: Path) -> DiskData:
        """Reads the npz data from the disk and transforms it properly"""
        if (key := (str(getattr(self, "name", None)), str(getattr(self, "normalization", None)), str(path))) in _CACHE:
            logger.debug2(f"HIT: '{key}'")
            return deepcopy(_CACHE[key])
        logger.debug2(f"MISS: '{key}'")
        data = np.load(path, allow_pickle=False)
        data = data if isinstance(data, np.ndarray) else data["arr_0"] # in case on npz, we need this as well
        data = data.astype(np.float32) if np.issubdtype(data.dtype, np.floating) else data # float16 is dangerous
        _CACHE[key] = deepcopy(data)
        return data

    @overrides
    def save_to_disk(self, memory_data: MemoryData, path: Path):
        """stores this item to the disk which can then be loaded via `load_from_disk`"""
        assert self.save_fn is not None
        self.save_fn(path, memory_data, allow_pickle=False) # pylint: disable=not-callable

    @overrides
    def memory_to_disk_fmt(self, memory_data: MemoryData) -> DiskData:
        return memory_data

    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        return MemoryData(disk_data)
