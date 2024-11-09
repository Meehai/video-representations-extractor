"""TaskMapper module"""
from pathlib import Path
from typing import Callable
from overrides import overrides
import numpy as np
import torch as tr

from vre.utils import VREVideo
from vre.representations import Representation, ComputeRepresentationMixin, ReprOut
from vre.stored_representation.npz_representation import NpzRepresentation

class TaskMapper(Representation, NpzRepresentation, ComputeRepresentationMixin):
    """TaskMapper implementation"""
    def __init__(self, name: str, merge_fn: Callable[[list[np.ndarray]], tr.Tensor], **kwargs):
        NpzRepresentation.__init__(self, name=name, **kwargs)
        Representation.__init__(self, name=name, dependencies=kwargs["dependencies"])
        ComputeRepresentationMixin.__init__(self)
        assert len(self.dependencies) > 0 and self.dep_names[0] != self.name, "Need at least one dependency"
        self.merge_fn = merge_fn

    @overrides
    def return_fn(self, load_data: tr.Tensor) -> tr.Tensor:
        """return_fn is the code that's ran between what's stored on the disk and what's actually sent to the model"""
        raise NotImplementedError(self)

    @overrides
    def load_from_disk(self, path: Path | list[Path]) -> tr.Tensor:
        paths = [path] if isinstance(path, Path) else path
        if self.dependencies == [self]: # this means it's already pre-computed and deps were updated in the Reader.
            res = super().load_from_disk(paths[0])
            return self.return_fn(res)
        dep_data = [dep.load_from_disk(path) for dep, path in zip(self.dependencies, paths)]
        res = self.merge_fn(dep_data)
        return self.return_fn(res)

    @overrides
    def make_images(self, video: VREVideo, ixs: list[int] | slice) -> np.ndarray:
        res = []
        for item in self.data.output:
            res.append(self.plot_fn(self.return_fn(tr.from_numpy(item))))
        return np.stack(res)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int] | slice):
        data = [dep.return_fn(tr.from_numpy(dep.data.output)) for dep in self.dependencies]
        res = []
        for i in range(len(data[0])):
            res.append(self.merge_fn([x[i] for x in data]))
        self.data = ReprOut(tr.stack(res).numpy(), key=ixs)
