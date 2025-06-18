"""TaskMapper module"""
from pathlib import Path
from abc import ABC, abstractmethod
from overrides import overrides
from vre_video import VREVideo

from .io_representation_mixin import IORepresentationMixin, MemoryData
from .representation import Representation, ReprOut

class TaskMapper(Representation, IORepresentationMixin, ABC):
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
        assert len(self.dependencies) > 0 and self.dep_names[0] != self.name, "Need at least one dependency"
        assert all(isinstance(dep, IORepresentationMixin) for dep in self.dependencies), self.dependencies
        self._n_channels = n_channels

    @property
    @overrides
    def n_channels(self) -> int:
        return self._n_channels

    @abstractmethod
    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        """merges all the dependencies (as MemoryData coming from their respective data) into a new MemoryData"""

    def compute_from_dependencies_paths(self, paths: Path | list[Path]) -> MemoryData:
        """used in MultiTaskReader. Unclear if it's a good idea"""
        paths = [paths] if isinstance(paths, Path) else paths
        assert isinstance(paths, list) and len(paths) == len(self.dependencies), (paths, self.dependencies)
        deps_memory_data = []
        dep: IORepresentationMixin
        for i, dep in enumerate(self.dependencies):
            deps_memory_data.append(dep.disk_to_memory_fmt(dep.load_from_disk(paths[i])))
        merged_data = self.merge_fn(deps_memory_data)
        assert isinstance(merged_data, MemoryData), (self, type(merged_data)) # TODO(!82): unittest this
        return merged_data

    @overrides
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        assert dep_data is not None, dep_data
        assert all(dep.key == ixs for dep in dep_data), ([dep.key for dep in dep_data], ixs)
        data = [dep.output for dep in dep_data]
        res = []
        for i in range(len(data[0])):
            res.append(self.merge_fn([x[i] for x in data]))
        assert all(isinstance(item, MemoryData) for item in res), (self, [type(item) for item in res])
        return ReprOut(video[ixs], MemoryData(res), key=ixs)
