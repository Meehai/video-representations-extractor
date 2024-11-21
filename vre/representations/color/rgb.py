"""RGB representation."""
from __future__ import annotations
import numpy as np
from overrides import overrides

from vre.utils import VREVideo, MemoryData
from vre.representations import (
    Representation, ReprOut, ComputeRepresentationMixin, NpIORepresentation, NormedRepresentationMixin)

class RGB(Representation, ComputeRepresentationMixin, NpIORepresentation, NormedRepresentationMixin):
    """Basic RGB representation."""
    def __init__(self, *args, **kwargs):
        Representation.__init__(self, *args, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)
        assert len(self.dependencies) == 0, self.dependencies
        self.output_dtype = "uint8"

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        self.data = ReprOut(frames=video[ixs], output=MemoryData(video[ixs]), key=ixs) # video[ixs] is cached

    @overrides
    def make_images(self) -> np.ndarray:
        y = self.unnormalize(self.data.output) if self.normalization is not None else self.data.output
        return y.astype(np.uint8)

    @property
    @overrides
    def n_channels(self) -> int:
        return 3
