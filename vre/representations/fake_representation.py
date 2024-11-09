"""FakeRepresentation module"""
from __future__ import annotations
import numpy as np
from overrides import overrides

from vre.utils import VREVideo
from .representation import Representation, ReprOut
from .compute_representation_mixin import ComputeRepresentationMixin

class FakeRepresentation(Representation, ComputeRepresentationMixin):
    """FakeRepresentation that is used in unit tests and some basic classes, like RGB."""
    def __init__(self, *args, **kwargs):
        Representation.__init__(self, *args, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        assert len(self.dependencies) == 0, self.dependencies

    @overrides
    def compute(self, video: VREVideo, ixs: list[int] | slice):
        assert self.data is None, "data must not be computed before calling this"
        self.data = ReprOut(output=np.array(video[ixs]), key=ixs)

    @overrides
    def make_images(self, video: VREVideo, ixs: list[int] | slice) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        return self.data.output
