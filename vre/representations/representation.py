"""VRE Representation module"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..utils import parsed_str_type, VREVideo
from ..logger import vre_logger as logger

@dataclass
class ReprOut:
    """The output of representation.make()"""
    output: np.ndarray
    extra: list[dict] | None = None

class Representation(ABC):
    """Generic Representation class for VRE"""
    def __init__(self, name: str, dependencies: list[Representation]):
        super().__init__()
        assert isinstance(dependencies, (set, list))
        self.name = name
        self.dependencies = dependencies

        self.batch_size: int | None = None
        self.output_size: tuple[int, int] | str | None = None
        self.video: VREVideo | None = None
        self.output_dir: Path | None = None


    @abstractmethod
    def make(self, frames: np.ndarray, dep_data: dict[str, ReprOut] | None = None) -> ReprOut:
        """
        Main method of this representation. Calls the internal representation's logic to transform the current provided
        RGB frame of the attached video into the output representation.
        Note: The representation that is returned is guaranteed to be a float32 (or uint8) numpy array.

        The returned value is either a simple numpy array of the same shape as the video plus an optional tuple with
        extra stuff. This extra stuff is whatever that representation may want to store about the frames it was given.

        This is also invoked for repr[t] and repr(t).
        """

    @abstractmethod
    def make_images(self, frames: np.ndarray, repr_data: ReprOut) -> np.ndarray:
        """Given the output of self.make(frames) of type ReprOut, return a [0:255] image for each frame"""

    @abstractmethod
    def resize(self, repr_data: ReprOut, new_size: tuple[int, int]) -> ReprOut:
        """
        Resizes the output of a self.make(frames) call into some other resolution
        Parameters:
        - repr_data The original representation output
        - new_size A tuple of two positive integers representing the new size
        Returns: A new representation output at the desired size
        """

    @abstractmethod
    def size(self, repr_data: ReprOut) -> tuple[int, int]:
        """Returns the (h, w) tuple of the size of the current representation"""

    ## Public methods ##

    def vre_dep_data(self, ix: slice) -> dict[str, ReprOut]:
        """iteratively collects all the dependencies needed by this representation"""
        assert self.video is not None, f"[{self}] self.video must be set before calling this"
        return {dep.name: dep.vre_make(ix) for dep in self.dependencies}

    def vre_make(self, ix: slice) -> ReprOut:
        """wrapper on top of make() that is ran in VRE context."""
        assert self.video is not None, f"[{self}] self.video must be set before calling this"
        if self.output_dir is not None:
            if (loaded_output := self._load_from_disk_if_possible(ix)) is not None:
                return loaded_output
        frames, dep_data = np.array(self.video[ix]), self.vre_dep_data(ix)
        res = self.make(frames, dep_data)
        assert isinstance(res, ReprOut), f"[{self}] Expected make() to produce ReprOut, got {type(res)})"
        assert not isinstance(res.output, ReprOut), f"[{self}] Recursive ReprOuts are not allowed"
        return res

    ## Private methods ##
    def _load_from_disk_if_possible(self, ix: slice) -> ReprOut | None:
        npy_paths: list[Path] = [self.output_dir / self.name / f"npy/{i}.npz" for i in range(ix.start, ix.stop)]
        extra_paths: list[Path] = [self.output_dir / self.name / f"npy/{i}_extra.npz"
                                    for i in range(ix.start, ix.stop)]
        if any(not x.exists() for x in npy_paths): # partial batches are considered 'not existing' and overwritten
            return None
        extras_exist = [x.exists() for x in extra_paths]
        assert (ee := sum(extras_exist)) in (0, (ep := len(extra_paths))), f"Found {ee}. Expected either 0 or {ep}"
        extra = [np.load(x, allow_pickle=True)["arr_0"].item() for x in extra_paths] if ee == ep else None
        logger.debug2(f"[{self}] Slice: [{ix.start}:{ix.stop - 1}]. All data found on disk and loaded")
        data = np.stack([np.load(x)["arr_0"] for x in npy_paths])
        return ReprOut(output=data, extra=extra)

    ## Magic methods ##

    def __getitem__(self, *args) -> ReprOut:
        raise NotImplementedError("Use self.__call__(args). __getitem__ doesn't make sense because of dependencies")

    def __call__(self, *args, **kwargs) -> ReprOut:
        return self.make(*args, **kwargs)

    def __repr__(self):
        return f"[Representation] {parsed_str_type(self)}({self.name})"
