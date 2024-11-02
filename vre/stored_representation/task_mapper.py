"""TaskMapper module"""
from pathlib import Path
from typing import Callable
import numpy as np
import torch as tr

from .stored_represntation import NpzRepresentation

class TaskMapper(NpzRepresentation):
    """TaskMapper implementation"""
    def __init__(self, *args, merge_fn: Callable[[list[np.ndarray]], tr.Tensor], **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.dependencies) > 0 and self.dep_names[0] != self.name, "Need at least one dependency"
        self.merge_fn = merge_fn

    def return_fn(self, load_data: tr.Tensor) -> tr.Tensor:
        """return_fn is the code that's ran between what's stored on the disk and what's actually sent to the model"""
        raise NotImplementedError

    def load_from_disk(self, path: Path | list[Path]) -> tr.Tensor:
        paths = [path] if isinstance(path, Path) else path
        if self.dependencies == [self]: # this means it's already pre-computed and deps were updated in the Reader.
            return self.return_fn(super().load_from_disk(paths[0]))
        dep_data = [dep.load_from_disk(path) for dep, path in zip(self.dependencies, paths)]
        return self.return_fn(self.merge_fn(dep_data))
