"""Computer Vision stored representations"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch as tr
import flow_vis
from overrides import overrides
from matplotlib.cm import Spectral # pylint: disable=no-name-in-module
from torch.nn import functional as F

from ..representations.hsv import rgb2hsv
from ..utils import colorize_semantic_segmentation
from .stored_represntation import NpzRepresentation
from .normed_representation import NormedRepresentation

class ColorRepresentation(NpzRepresentation, NormedRepresentation):
    """ColorRepresentation -- a wrapper over all 3-channeled colored representations"""
    def __init__(self, name: str, dependencies: list[str] | None = None):
        NpzRepresentation.__init__(self, name, n_channels=3, dependencies=dependencies)
        NormedRepresentation.__init__(self)

    @overrides
    def plot_fn(self, x: tr.Tensor) -> np.ndarray:
        x = self.unnormalize(x.detach().cpu()) if self.normalization is not None else x.detach().cpu()
        return x.byte().numpy()

class RGBRepresentation(ColorRepresentation): pass # pylint: disable=missing-class-docstring, multiple-statements

class NormalsRepresentation(ColorRepresentation):
    """NormalsRepresentation -- CV representation for world and camera normals"""
    @overrides
    def plot_fn(self, x: tr.Tensor) -> np.ndarray:
        x = self.unnormalize(x.detach().cpu()) if self.normalization is not None else x.detach().cpu()
        return (x * 255).byte().numpy()

class HSVRepresentation(ColorRepresentation):
    """HSVRepresentation -- CV representation for HSV derived from RGB directly"""
    @overrides
    def load_from_disk(self, path: Path) -> tr.Tensor:
        rgb = NpzRepresentation.load_from_disk(self, path)
        return tr.from_numpy(rgb2hsv(rgb.numpy())).float()

    @overrides
    def plot_fn(self, x: tr.Tensor) -> np.ndarray:
        x = self.unnormalize(x.detach().cpu()) if self.normalization is not None else x.detach().cpu()
        return (x * 255).byte().numpy()

class EdgesRepresentation(NpzRepresentation, NormedRepresentation):
    """EdgesRepresentation -- CV representation for 1-channeled edges/boundaries"""
    def __init__(self, *args, **kwargs):
        NpzRepresentation.__init__(self, *args, n_channels=1, **kwargs)
        NormedRepresentation.__init__(self)

    def plot_fn(self, x: tr.Tensor) -> np.ndarray:
        x = self.unnormalize(x.detach().cpu()) if self.normalization is not None else x.detach().cpu()
        return (x.repeat(1, 1, 3) * 255).byte().numpy()

class DepthRepresentation(NpzRepresentation, NormedRepresentation):
    """DepthRepresentation. Implements depth task-specific stuff, like spectral map for plots."""
    def __init__(self, name: str, min_depth: float, max_depth: float, **kwargs):
        NpzRepresentation.__init__(self, name, n_channels=1, **kwargs)
        NormedRepresentation.__init__(self)

        self.min_depth = min_depth
        self.max_depth = max_depth

    @overrides
    def load_from_disk(self, path: Path) -> tr.Tensor:
        """Reads the npz data from the disk and transforms it properly"""
        res = super().load_from_disk(path)
        res_clip = res.clip(self.min_depth, self.max_depth)
        return res_clip

    @overrides
    def plot_fn(self, x: tr.Tensor) -> np.ndarray:
        x = x.detach().clip(0, 1).squeeze().cpu().numpy()
        _min, _max = np.percentile(x, [1, 95])
        x = np.nan_to_num((x - _min) / (_max - _min), False, 0, 0, 0).clip(0, 1)
        y: np.ndarray = Spectral(x)[..., 0:3] * 255
        return y.astype(np.uint8)

class OpticalFlowRepresentation(NpzRepresentation, NormedRepresentation):
    """OpticalFlowRepresentation. Implements flow task-specific stuff, like using flow_vis."""
    def __init__(self, *args, **kwargs):
        NpzRepresentation.__init__(self, *args, n_channels=2, **kwargs)
        NormedRepresentation.__init__(self)

    @overrides
    def plot_fn(self, x: tr.Tensor) -> np.ndarray:
        _min, _max = x.min(0)[0].min(0)[0], x.max(0)[0].max(0)[0]
        x = ((x - _min) / (_max - _min)).nan_to_num(0, 0, 0).detach().cpu().numpy()
        return flow_vis.flow_to_color(x)

class SemanticRepresentation(NpzRepresentation):
    """SemanticRepresentation. Implements semantic task-specific stuff, like argmaxing if needed"""
    def __init__(self, *args, classes: int | list[str], color_map: list[tuple[int, int, int]], **kwargs):
        self.n_classes = len(list(range(classes)) if isinstance(classes, int) else classes)
        super().__init__(*args, **kwargs, n_channels=self.n_classes)
        self.classes = list(range(classes)) if isinstance(classes, int) else classes
        self.color_map = color_map
        assert len(color_map) == self.n_classes and self.n_classes > 1, (color_map, self.n_classes)

    @overrides
    def load_from_disk(self, path: Path) -> tr.Tensor:
        res = super().load_from_disk(path)
        if len(res.shape) == 3:
            assert res.shape[-1] == self.n_classes, f"Expected {self.n_classes} (HxWxC), got {res.shape[-1]}"
            res = res.argmax(-1)
        assert len(res.shape) == 2, f"Only argmaxed data supported, got: {res.shape}"
        res = F.one_hot(res.long(), num_classes=self.n_classes).float() # pylint: disable=not-callable
        return res

    @overrides
    def plot_fn(self, x: tr.Tensor) -> np.ndarray:
        x_argmax = x.squeeze().nan_to_num(0).detach().argmax(-1).cpu().numpy()
        return colorize_semantic_segmentation(x_argmax, self.classes, self.color_map,
                                              font_size_scale=2, original_rgb=None)
