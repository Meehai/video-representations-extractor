"""RGB representation."""
from __future__ import annotations
from overrides import overrides

from vre.utils import MemoryData, ReprOut
from vre.vre_video import VREVideo
from vre.representations import ComputeRepresentationMixin

from .color_representation import ColorRepresentation

class RGB(ColorRepresentation, ComputeRepresentationMixin):
    """Basic RGB representation."""
    def __init__(self, *args, **kwargs):
        ColorRepresentation.__init__(self, *args, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        assert len(self.dependencies) == 0, self.dependencies
        self.output_dtype = "uint8"

    @overrides
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        return ReprOut(frames=video[ixs], output=MemoryData(video[ixs]), key=ixs) # video[ixs] is cached
