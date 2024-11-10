"""TaskMapper module"""
from pathlib import Path
from abc import ABC, abstractmethod
from overrides import overrides
import numpy as np
import torch as tr

from vre.utils import VREVideo
from .npz_representation import NpzRepresentation
from .representation import Representation, ReprOut
from .compute_representation_mixin import ComputeRepresentationMixin

class TaskMapper(Representation, NpzRepresentation, ComputeRepresentationMixin, ABC):
    """
    TaskMapper abstract class.
    Must implement: 'merge_fn', 'return_fn' (StoredRepresentation) and 'plot_fn' (Representation)
    """
    def __init__(self, name: str, **kwargs):
        Representation.__init__(self, name=name, dependencies=kwargs["dependencies"])
        NpzRepresentation.__init__(self, name=name, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        assert len(self.dependencies) > 0 and self.dep_names[0] != self.name, "Need at least one dependency"

    @abstractmethod
    def merge_fn(self, dep_data: list[np.ndarray]) -> tr.Tensor:
        """merges all the dependencies (as np.array coming from their .load_from_disk) into a tr.Tensor"""

    @overrides
    def load_from_disk(self, path: Path | list[Path]) -> tr.Tensor:
        # TODO: most likely this code should have 0 return_fn calls and leave this to the use code (and Reader)
        paths = [path] if isinstance(path, Path) else path
        if self.dependencies == [self]: # this means it's already pre-computed and deps were updated in the Reader.
            res = super().load_from_disk(paths[0])
            return self.return_fn(res)
        dep_data = [dep.load_from_disk(path) for dep, path in zip(self.dependencies, paths)]
        res = self.merge_fn(dep_data)
        return self.return_fn(res)

    @overrides
    def make_images(self, video: VREVideo, ixs: list[int] | slice) -> np.ndarray:
        # TODO: most likely this code should have 0 return_fn calls and leave this to the use code (and Reader)
        res = []
        for item in self.data.output:
            res.append(self.plot_fn(self.return_fn(tr.from_numpy(item))))
        return np.stack(res)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int] | slice):
        # TODO: most likely this code should have 0 return_fn calls and leave this to the use code (and Reader)
        # TODO: check (& make test) that the dep.data refers to the same keys, otherwise compute again!
        data = [dep.return_fn(tr.from_numpy(dep.data.output)) for dep in self.dependencies]
        res = []
        for i in range(len(data[0])):
            res.append(self.merge_fn([x[i] for x in data]))
        self.data = ReprOut(tr.stack(res).numpy(), key=ixs)
