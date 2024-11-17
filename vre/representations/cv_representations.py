"""Computer Vision stored representations"""
from __future__ import annotations
import numpy as np
import flow_vis
from overrides import overrides
from matplotlib.cm import Spectral # pylint: disable=no-name-in-module

from vre.utils import colorize_semantic_segmentation
from .np_io_representation import NpIORepresentation, DiskData, MemoryData
from .normed_representation_mixin import NormedRepresentationMixin
from .representation import Representation
from .color.hsv import rgb2hsv

class ColorRepresentation(Representation, NpIORepresentation, NormedRepresentationMixin):
    """ColorRepresentation -- a wrapper over all 3-channeled colored representations"""
    def __init__(self, name: str, dependencies: list[Representation] | None = None):
        Representation.__init__(self, name=name, dependencies=dependencies)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)
        self.n_channels = 3

    @overrides
    def make_images(self) -> np.ndarray:
        y = self.unnormalize(self.data.output) if self.normalization is not None else self.data.output
        return y.astype(np.uint8)

class RGBRepresentation(ColorRepresentation): pass # pylint: disable=missing-class-docstring, multiple-statements

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

class EdgesRepresentation(Representation, NpIORepresentation, NormedRepresentationMixin):
    """EdgesRepresentation -- CV representation for 1-channeled edges/boundaries"""
    def __init__(self, *args, **kwargs):
        Representation.__init__(self, *args, **kwargs)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)
        self.n_channels = 1 # TODO

    def make_images(self) -> np.ndarray:
        y = self.unnormalize(self.data.output) if self.normalization is not None else self.data.output
        return (np.repeat(y, 3, axis=-1) * 255).astype(np.uint8)

class DepthRepresentation(Representation, NpIORepresentation, NormedRepresentationMixin):
    """DepthRepresentation. Implements depth task-specific stuff, like spectral map for plots."""
    def __init__(self, name: str, min_depth: float, max_depth: float, **kwargs):
        Representation.__init__(self, name=name, **kwargs)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)
        self.n_channels = 1 # TODO
        self.min_depth = min_depth
        self.max_depth = max_depth

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

class OpticalFlowRepresentation(Representation, NpIORepresentation, NormedRepresentationMixin):
    """OpticalFlowRepresentation. Implements flow task-specific stuff, like using flow_vis."""
    def __init__(self, *args, **kwargs):
        Representation.__init__(self, *args, **kwargs)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)
        self.n_channels = 2 # TODO

    @overrides
    def make_images(self) -> np.ndarray:
        _min, _max = self.data.output.min(0).min(0), self.data.output.max(0).max(0)
        y = np.nan_to_num(((self.data.output - _min) / (_max - _min)), False, 0, 0, 0).astype(np.float32)
        return np.array([flow_vis.flow_to_color(_y) for _y in y])

class SemanticRepresentation(Representation, NpIORepresentation):
    """SemanticRepresentation. Implements semantic task-specific stuff, like argmaxing if needed"""
    def __init__(self, *args, classes: int | list[str], color_map: list[tuple[int, int, int]], **kwargs):
        self.n_classes = len(list(range(classes)) if isinstance(classes, int) else classes)
        Representation.__init__(self, *args, **kwargs)
        NpIORepresentation.__init__(self)
        self.classes = list(range(classes)) if isinstance(classes, int) else classes
        self.color_map = color_map
        self.n_channels = self.n_classes # TODO
        assert len(color_map) == self.n_classes and self.n_classes > 1, (color_map, self.n_classes)

    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        assert disk_data.dtype in (np.uint8, np.uint16), disk_data.dtype
        return MemoryData(np.eye(self.n_classes)[disk_data].astype(np.float32))

    @overrides
    def make_images(self) -> np.ndarray:
        res = [colorize_semantic_segmentation(item.argmax(-1).astype(int), self.classes, self.color_map,
                                              original_rgb=None, font_size_scale=2) for item in self.data.output]
        return np.array(res)
