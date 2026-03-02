"""depth_representationpy -- module implementing a Depth Represenatation generic class"""
from overrides import overrides
import numpy as np
from vre.utils import MemoryData, DiskData, colorize_depth
from vre.representations.mixins import NpIORepresentation, NormedRepresentationMixin, ResizableRepresentationMixin
from vre.representations import Representation, ReprOut

class DepthRepresentation(Representation, NpIORepresentation, NormedRepresentationMixin, ResizableRepresentationMixin):
    """DepthRepresentation. Implements depth task-specific stuff, like spectral map for plots."""
    def __init__(self, name: str, min_depth: float, max_depth: float, **kwargs):
        Representation.__init__(self, name, **kwargs)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)
        ResizableRepresentationMixin.__init__(self)
        self.min_depth = min_depth
        self.max_depth = max_depth

    @property
    @overrides
    def n_channels(self) -> int:
        return 1

    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        assert not isinstance(disk_data, MemoryData), type(disk_data)
        return MemoryData(disk_data.clip(self.min_depth, self.max_depth))

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        return (colorize_depth(data.output, percentiles=[1, 95]) * 255).astype(np.uint8)
