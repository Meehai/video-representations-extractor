"""repr_memory_layout.py implements the data structures for holding Representation Outputs (ReprOut) and MemoryData."""
from __future__ import annotations
import numpy as np
from .lovely import lo

DiskData = np.ndarray

class MemoryData(np.ndarray):
    """Wrapper on top of ndarray to differentiate between DiskData and MemoryData."""
    def __new__(cls, *args, **kwargs):
        # Create a new instance of MemoryData using np.array's functionality
        obj = np.asarray(*args, **kwargs).view(cls)
        return obj

    @property
    def v(self) -> str:
        """returns the underlying numpy array string in case we want a more verbose output"""
        return np.ndarray.__str__(self)

    def __repr__(self):
        return lo(self)
