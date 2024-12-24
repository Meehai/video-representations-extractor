"""Computer Vision stored representations"""
from __future__ import annotations
import numpy as np
from overrides import overrides

from vre.utils import VREVideo, colorize_optical_flow, colorize_depth, ReprOut
from .np_io_representation import NpIORepresentation, DiskData, MemoryData
from .normed_representation_mixin import NormedRepresentationMixin
from .representation import Representation
from .compute_representation_mixin import ComputeRepresentationMixin
from .color.hsv import rgb2hsv

class ExternalRepresentation(Representation, ComputeRepresentationMixin, NormedRepresentationMixin):
    """External Representations wrapper, so we can load stuff on the disk w/o explicit representations"""
    def __init__(self, name, dependencies = None):
        Representation.__init__(self, name=name, dependencies=dependencies)
        ComputeRepresentationMixin.__init__(self)
        NormedRepresentationMixin.__init__(self)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        raise NotImplementedError(f"{self} supposed to be used only as external representation")

class ColorRepresentation(ExternalRepresentation, NpIORepresentation):
    """ColorRepresentation -- a wrapper over all 3-channeled colored representations"""
    def __init__(self, name: str, dependencies: list[Representation] | None = None):
        ExternalRepresentation.__init__(self, name=name, dependencies=dependencies)
        NpIORepresentation.__init__(self)

    @property
    @overrides
    def n_channels(self) -> int:
        return 3

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        y = self.unnormalize(data.output) if self.normalization is not None else self.data.output
        return y.astype(np.uint8)

class NormalsRepresentation(ColorRepresentation):
    """NormalsRepresentation -- CV representation for world and camera normals"""
    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        y = self.unnormalize(data.output) if self.normalization is not None else self.data.output
        return (y * 255).astype(np.uint8)

class HSVRepresentation(ColorRepresentation):
    """HSVRepresentation -- CV representation for HSV derived from RGB directly"""
    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        return MemoryData(rgb2hsv(disk_data))

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        y = self.unnormalize(data.output) if self.normalization is not None else self.data.output
        return (y * 255).astype(np.uint8)

class EdgesRepresentation(ExternalRepresentation, NpIORepresentation):
    """EdgesRepresentation -- CV representation for 1-channeled edges/boundaries"""
    def __init__(self, *args, **kwargs):
        ExternalRepresentation.__init__(self, *args, **kwargs)
        NpIORepresentation.__init__(self)

    @property
    @overrides
    def n_channels(self) -> int:
        return 1

    def make_images(self, data: ReprOut) -> np.ndarray:
        y = self.unnormalize(data.output) if self.normalization is not None else self.data.output
        return (np.repeat(y, 3, axis=-1) * 255).astype(np.uint8)

class DepthRepresentation(ExternalRepresentation, NpIORepresentation):
    """DepthRepresentation. Implements depth task-specific stuff, like spectral map for plots."""
    def __init__(self, name: str, min_depth: float, max_depth: float, **kwargs):
        ExternalRepresentation.__init__(self, name=name, **kwargs)
        NpIORepresentation.__init__(self)
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

class OpticalFlowRepresentation(ExternalRepresentation, NpIORepresentation):
    """OpticalFlowRepresentation. Implements flow task-specific stuff."""
    def __init__(self, *args, **kwargs):
        ExternalRepresentation.__init__(self, *args, **kwargs)
        NpIORepresentation.__init__(self)

    @property
    @overrides
    def n_channels(self) -> int:
        return 2

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        _min, _max = self.data.output.min(0).min(0), self.data.output.max(0).max(0)
        y = np.nan_to_num(((self.data.output - _min) / (_max - _min)), False, 0, 0, 0).astype(np.float32)
        return colorize_optical_flow(y)
