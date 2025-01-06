"""depth_representationpy -- module implementing a Depth Represenatation generic class"""
from overrides import overrides
import numpy as np
from vre.utils import MemoryData, DiskData, colorize_depth, ReprOut, VREVideo
from vre.representations import (
    Representation, NpIORepresentation, ComputeRepresentationMixin, NormedRepresentationMixin)

class DepthRepresentation(Representation, NpIORepresentation, ComputeRepresentationMixin, NormedRepresentationMixin):
    """DepthRepresentation. Implements depth task-specific stuff, like spectral map for plots."""
    def __init__(self, name: str, min_depth: float, max_depth: float, **kwargs):
        ComputeRepresentationMixin.__init__(self)
        NormedRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        Representation.__init__(self, name, **kwargs)
        self.min_depth = min_depth
        self.max_depth = max_depth

    @property
    @overrides
    def n_channels(self) -> int:
        return 1

    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        return MemoryData(disk_data.clip(self.min_depth, self.max_depth))

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        return (colorize_depth(self.data.output, percentiles=[1, 95]) * 255).astype(np.uint8)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        raise NotImplementedError(f"[{self}] compute() must be overriden. We inherit it for output_dtype/size etc.")
