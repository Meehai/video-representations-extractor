"""TaskMapper module"""
from pathlib import Path
from abc import ABC, abstractmethod
from overrides import overrides
import numpy as np

from vre.utils import VREVideo
from .io_representation_mixin import IORepresentationMixin, MemoryData
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
        assert all(isinstance(dep, IORepresentationMixin) for dep in self.dependencies), self.dependencies
        self.n_channels = n_channels

    @abstractmethod
    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        """merges all the dependencies (as MemoryData coming from their respective data) into a new MemoryData"""

    def compute_from_dependencies_paths(self, paths: Path | list[Path]) -> MemoryData:
        """used in MultiTaskReader. Unclear if it's a good idea"""
        paths = [paths] if isinstance(paths, Path) else paths
        assert isinstance(paths, list) and len(paths) == len(self.dependencies), (paths, self.dependencies)
        dep_disk_data = [dep.load_from_disk(path) for dep, path in zip(self.dependencies, paths)]
        dep_memory_data = [dep.from_disk_fmt(disk_data) for dep, disk_data in zip(self.dependencies, dep_disk_data)]
        merged_data = self.merge_fn(dep_memory_data)
        return merged_data

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        data = [dep.data.output for dep in self.dependencies]
        assert all(dep.data.key == ixs for dep in self.dependencies), ([dep.data.key for dep in self.dependencies], ixs)
        res = []
        for i in range(len(data[0])):
            res.append(self.merge_fn([x[i] for x in data]))
        self.data = ReprOut(np.array(video[ixs]), np.array(res), key=ixs)
