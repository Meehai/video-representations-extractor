"""FakeRepresentation module for tests"""
from __future__ import annotations
import numpy as np
from overrides import overrides

from vre.logger import vre_logger as logger
from vre.utils import VREVideo, MemoryData
from vre.representations import Representation, ReprOut, ComputeRepresentationMixin, NpIORepresentation

class FakeRepresentation(Representation, ComputeRepresentationMixin, NpIORepresentation):
    """FakeRepresentation that is used in unit tests and some basic classes, like RGB."""
    def __init__(self, *args, **kwargs):
        Representation.__init__(self, *args, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        if self.dependencies != 0:
            logger.warning(f"{self} has {len(self.dependencies)} dependencies. Usually it's supposed to be 0")

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, "data must not be computed before calling this"
        self.data = ReprOut(frames=video[ixs], output=MemoryData(video[ixs]), key=ixs)

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        assert data.output is not None, f"[{self}] data must be first computed using compute(), got {data}"
        return data.output

    @property
    @overrides
    def n_channels(self) -> int:
        raise ValueError("I hate inheritance. Makes no sense for this representation")
