"""RGB representation."""
from __future__ import annotations
from overrides import overrides

from vre.utils import VREVideo, MemoryData, ReprOut
from .color_representation import ColorRepresentation

class RGB(ColorRepresentation):
    """Basic RGB representation."""
    def __init__(self, *args, **kwargs):
        ColorRepresentation.__init__(self, *args, **kwargs)
        assert len(self.dependencies) == 0, self.dependencies
        self.output_dtype = "uint8"

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        self.data = ReprOut(frames=video[ixs], output=MemoryData(video[ixs]), key=ixs) # video[ixs] is cached
