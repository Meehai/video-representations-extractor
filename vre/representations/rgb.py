"""RGB representation. Simply inherit FakeRepresentation that does all this needs (i.e. copy pasta the frames)"""
from __future__ import annotations
import numpy as np
from overrides import overrides

from vre.utils import VREVideo
from vre.representations import Representation, ReprOut, ComputeRepresentationMixin

class RGB(Representation, ComputeRepresentationMixin):
    """FakeRepresentation that is used in unit tests and some basic classes, like RGB."""
    def __init__(self, *args, **kwargs):
        Representation.__init__(self, *args, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        assert len(self.dependencies) == 0, self.dependencies
        self.output_dtype = "uint8"

    @overrides
    def compute(self, video: VREVideo, ixs: list[int] | slice):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        self.data = ReprOut(output=np.array(video[ixs]), key=ixs)

    @overrides
    def make_images(self, video: VREVideo, ixs: list[int] | slice) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        return self.data.output
