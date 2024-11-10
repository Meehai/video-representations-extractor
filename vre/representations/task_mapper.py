"""TaskMapper module"""
from pathlib import Path
from abc import ABC, abstractmethod
from overrides import overrides
import numpy as np

from vre.utils import VREVideo
from .io_representation_mixin import IORepresentationMixin#, DiskData, MemoryData
from .representation import Representation, ReprOut
from .compute_representation_mixin import ComputeRepresentationMixin

class TaskMapper(Representation, IORepresentationMixin, ComputeRepresentationMixin, ABC):
    """
    TaskMapper abstract class for >=1 dependency that is transformed (mapped) (e.g. mapillary -> 8 classes).
    Must implement:
    - `merge_fn` - the actual logic that merges all the dependencies into a single one
    - `Representation`: 'make_images'
    - `IORepresentationMixin`: `save_to_disk`
    """
    def __init__(self, name: str, n_channels: int, **kwargs):
        Representation.__init__(self, name=name, dependencies=kwargs["dependencies"])
        IORepresentationMixin.__init__(self)
        ComputeRepresentationMixin.__init__(self)
        assert len(self.dependencies) > 0 and self.dep_names[0] != self.name, "Need at least one dependency"
        self.n_channels = n_channels

    @abstractmethod
    def merge_fn(self, dep_data: list[np.ndarray]) -> np.ndarray:
        """merges all the dependencies (as np.array coming from their .load_from_disk) into a tr.Tensor"""

    @overrides(check_signature=False)
    def load_from_disk(self, path: Path | list[Path]) -> np.ndarray:
        # TODO: most likely this code should have 0 return_fn calls and leave this to the use code (and Reader)
        raise NotImplementedError("TODO")
        # paths = [path] if isinstance(path, Path) else path
        # if self.dependencies == [self]: # this means it's already pre-computed and deps were updated in the Reader.
        #     breakpoint()
        #     res = super().load_from_disk(paths[0])
        #     return self.return_fn(res)
        # breakpoint()
        # dep_data = []
        # for dep, path in zip(self.dependencies, paths):
        #     assert isinstance(dep, IORepresentationMixin), (dep, type(dep)) # TODO: what about compute here? good ?
        #     dep_data.append(dep.load_from_disk(path))
        # res = self.merge_fn(dep_data)
        # return self.return_fn(res)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int] | slice):
        # TODO: check (& make test) that the dep.data refers to the same keys, otherwise compute again!
        data = [dep.data.output for dep in self.dependencies]
        assert all(dep.data.key == ixs for dep in self.dependencies), ([dep.data.key for dep in self.dependencies], ixs)
        res = []
        for i in range(len(data[0])):
            res.append(self.merge_fn([x[i] for x in data]))
        self.data = ReprOut(np.array(video[ixs]), np.array(res), key=ixs)
