"""repr_memory_layout.py implements the data structures for holding Representation Outputs (ReprOut) and MemoryData."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .lovely import lo

DiskData = np.ndarray

class MemoryData(np.ndarray):
    """Wrapper on top of ndarray to differentiate between DiskData and MemoryData."""
    def __new__(cls, *args, **kwargs):
        # Create a new instance of MemoryData using np.array's functionality
        obj = np.asarray(*args, **kwargs).view(cls)
        return obj

    def __repr__(self):
        return lo(self)

@dataclass
class ReprOut:
    """The output of representation.compute()"""
    frames: np.ndarray | None
    output: MemoryData
    key: list[int]
    extra: list[dict] | None = None
    output_images: np.ndarray | None = None

    def __post_init__(self):
        assert isinstance(self.output, MemoryData), f"Use MemoryData(arr) if array. Got: {type(self.output)}"
        assert len(self.output) == len(self.key), (len(self.output), len(self.key))
        assert self.frames is None or len(self.frames) == len(self.output), (len(self.frames), len(self.output))

    def __repr__(self):
        return (f"[ReprOut](key={self.key}, output={self.output}, output_images={lo(self.output_images)} "
                f"extra set={self.extra is not None}, frames={lo(self.frames)})")
