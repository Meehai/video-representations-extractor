"""VRE Representation module"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from lovely_numpy import lo
import numpy as np

from ..utils import parsed_str_type, image_resize_batch

@dataclass
class ReprOut:
    """The output of representation.compute()"""
    frames: np.ndarray
    output: np.ndarray
    key: list[int]
    extra: list[dict] | None = None

    def __post_init__(self):
        assert isinstance(self.output, np.ndarray), type(self.output)
        if isinstance(self.key, slice):
            self.key = list(range(self.key.start, self.key.stop))
        assert len(self.output) == len(self.key), (len(self.output), len(self.key))

    def __repr__(self):
        return f"[ReprOut](output={lo(self.output)}, key={self.key}, extra={self.extra})"

class Representation(ABC):
    """Generic Representation class for VRE"""
    def __init__(self, name: str, dependencies: list[Representation] | None = None):
        dependencies = [] if dependencies is None else dependencies
        assert isinstance(dependencies, list), type(dependencies)
        assert all(isinstance(dep, Representation) for dep in dependencies), (type(dep) for dep in dependencies)
        self.name = name
        self.dependencies = dependencies
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
        """Returns the (h, w) tuple of the size of the current representation"""
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        return tuple(self.data.output.shape)

    @property
    def is_classification(self) -> bool:
        """if we have self.classes. Used in MultiTaskReader."""
        return hasattr(self, "classes") and self.classes is not None # pylint: disable=no-member

    def plot_fn(self, x: np.ndarray | "tr.Tensor") -> np.ndarray:
        """hack"""
        x = x.cpu().numpy() if hasattr(x, "numpy") else x
        self.data = ReprOut(None, x[None], [0])
        return self.make_images()[0]

    def resize(self, new_size: tuple[int, int]):
        """resizes the data. size is provided in (h, w)"""
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        interpolation = "nearest" if np.issubdtype(d := self.data.output.dtype, np.integer) or d == bool else "bilinear"
        self.data = ReprOut(frames=self.data.frames, key=self.data.key, extra=self.data.extra,
                            output=image_resize_batch(self.data.output, *new_size, interpolation=interpolation))

    def cast(self, dtype: str):
        """Cast the output of a self.compute(frames) call into some other dtype"""
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        if (np.issubdtype(self.data.output.dtype, np.integer) and np.issubdtype(dtype, np.floating) or
            np.issubdtype(self.data.output.dtype, np.floating) and np.issubdtype(dtype, np.integer)):
            raise TypeError(f"Cannot convert {self.data.output.dtype} to {dtype}")
        self.data = ReprOut(frames=self.data.frames, output=self.data.output.astype(dtype),
                            extra=self.data.extra, key=self.data.key)

    ## Magic methods ##
    def __repr__(self):
        return f"{parsed_str_type(self)}({self.name}{f' {self.dep_names}' if len(self.dep_names) > 0 else ''})"
