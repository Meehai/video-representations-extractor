
"""color_representation.py -- module implementing an 'Color' (RGB-like) Represenatation generic class"""
from overrides import overrides
import numpy as np
from vre.representations import (
    Representation, NpIORepresentation, ComputeRepresentationMixin, NormedRepresentationMixin)
from vre.utils import ReprOut, VREVideo

class ColorRepresentation(Representation, ComputeRepresentationMixin, NormedRepresentationMixin, NpIORepresentation):
    """ColorRepresentation -- a wrapper over all 3-channeled colored representations"""
    def __init__(self, name: str, **kwargs):
        Representation.__init__(self, name, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)

    @property
    @overrides
    def n_channels(self) -> int:
        return 3

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        y = self.unnormalize(data.output) if self.normalization is not None else self.data.output
        return y.astype(np.uint8)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        raise NotImplementedError(f"[{self}] compute() must be overriden. We inherit it for output_dtype/size etc.")
