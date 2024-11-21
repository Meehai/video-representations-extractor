"""Computer Vision stored representations"""
from __future__ import annotations
import numpy as np
import flow_vis
from overrides import overrides
from matplotlib.cm import Spectral # pylint: disable=no-name-in-module

from vre.utils import colorize_semantic_segmentation, VREVideo
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
    def make_images(self) -> np.ndarray:
        y = self.unnormalize(self.data.output) if self.normalization is not None else self.data.output
        return y.astype(np.uint8)

class NormalsRepresentation(ColorRepresentation):
    """NormalsRepresentation -- CV representation for world and camera normals"""
    @overrides
    def make_images(self) -> np.ndarray:
        y = self.unnormalize(self.data.output) if self.normalization is not None else self.data.output
        return (y * 255).astype(np.uint8)

class HSVRepresentation(ColorRepresentation):
    """HSVRepresentation -- CV representation for HSV derived from RGB directly"""
    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        return MemoryData(rgb2hsv(disk_data))

    @overrides
    def make_images(self) -> np.ndarray:
        y = self.unnormalize(self.data.output) if self.normalization is not None else self.data.output
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

    def make_images(self) -> np.ndarray:
        y = self.unnormalize(self.data.output) if self.normalization is not None else self.data.output
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
    def make_images(self) -> np.ndarray:
        x = self.data.output.clip(0, 1)
        x = x[..., 0] if x.shape[-1] == 1 else x
        _min, _max = np.percentile(x, [1, 95])
        x = np.nan_to_num((x - _min) / (_max - _min), False, 0, 0, 0).clip(0, 1)
        y: np.ndarray = Spectral(x)[..., 0:3] * 255
        return y.astype(np.uint8)

class OpticalFlowRepresentation(ExternalRepresentation, NpIORepresentation):
    """OpticalFlowRepresentation. Implements flow task-specific stuff, like using flow_vis."""
    def __init__(self, *args, **kwargs):
        ExternalRepresentation.__init__(self, *args, **kwargs)
        NpIORepresentation.__init__(self)

    @property
    @overrides
    def n_channels(self) -> int:
        return 2

    @overrides
    def make_images(self) -> np.ndarray:
        _min, _max = self.data.output.min(0).min(0), self.data.output.max(0).max(0)
        y = np.nan_to_num(((self.data.output - _min) / (_max - _min)), False, 0, 0, 0).astype(np.float32)
        return np.array([flow_vis.flow_to_color(_y) for _y in y])

class SemanticRepresentation(Representation, ComputeRepresentationMixin, NpIORepresentation):
    """SemanticRepresentation. Implements semantic task-specific stuff, like argmaxing if needed"""
    def __init__(self, *args, classes: int | list[str], color_map: list[tuple[int, int, int]], **kwargs):
        self.n_classes = len(list(range(classes)) if isinstance(classes, int) else classes)
        Representation.__init__(self, *args, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        self.classes = list(range(classes)) if isinstance(classes, int) else classes
        self.color_map = color_map
        assert len(color_map) == self.n_classes and self.n_classes > 1, (color_map, self.n_classes)

    @property
    @overrides
    def n_channels(self) -> int:
        return self.n_classes

    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        assert disk_data.dtype in (np.uint8, np.uint16), disk_data.dtype
        return MemoryData(np.eye(self.n_classes)[disk_data].astype(np.float32))

    @overrides
    def make_images(self) -> np.ndarray:
        return colorize_semantic_segmentation(self.data.output.argmax(-1), self.classes, self.color_map)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        raise NotImplementedError(f"{self} supposed to be used only as external representation")
