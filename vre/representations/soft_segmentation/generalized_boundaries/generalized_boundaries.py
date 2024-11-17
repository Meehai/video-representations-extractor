"""Generalized boundaries (softseg) representation"""
import torch as tr
import numpy as np
from overrides import overrides

from vre.utils import VREVideo, MemoryData
from vre.representations import Representation, ReprOut, ComputeRepresentationMixin, NpIORepresentation
from vre.representations.soft_segmentation.generalized_boundaries.gb_impl.softseg import soft_seg

class GeneralizedBoundaries(Representation, ComputeRepresentationMixin, NpIORepresentation):
    """
    Soft-seg implementation from https://link.springer.com/chapter/10.1007/978-3-642-33765-9_37
    Parameters:
    - use_median_filtering: Apply a median filtering postprocessing pass.
    - adjust_to_rgb: Return a RGB soft segmentation image in a similar colormap as the input.
    - max_channels: Max segmentation maps. Upper bounded at ~60.
    """
    def __init__(self, use_median_filtering: bool, adjust_to_rgb: bool, max_channels: int, **kwargs):
        Representation.__init__(self, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        self.use_median_filtering = use_median_filtering
        self.adjust_to_rgb = adjust_to_rgb
        self.max_channels = max_channels

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        x = tr.from_numpy(video[ixs]).type(tr.float) / 255
        x = x.permute(0, 3, 1, 2)
        y = soft_seg(x, use_median_filtering=self.use_median_filtering, as_image=self.adjust_to_rgb,
                     max_channels=self.max_channels)
        y = y.permute(0, 2, 3, 1).cpu().numpy()
        self.data = ReprOut(frames=video[ixs], output=MemoryData(y), key=ixs)

    @overrides
    def make_images(self) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        return (self.data.output * 255).astype(np.uint8)
