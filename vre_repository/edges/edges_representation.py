"""edges_representation.py -- module implementing an Edges Represenatation generic class"""
from overrides import overrides
import numpy as np
from vre_video import VREVideo
from vre.utils import ReprOut
from vre.representations import (
    Representation, NpIORepresentation, NormedRepresentationMixin)

class EdgesRepresentation(Representation, NpIORepresentation, NormedRepresentationMixin):
    """EdgesRepresentation -- CV representation for 1-channeled edges/boundaries"""
    def __init__(self, name: str, **kwargs):
        Representation.__init__(self, name, **kwargs)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)

    @property
    @overrides
    def n_channels(self) -> int:
        return 1

    def make_images(self, data: ReprOut) -> np.ndarray:
        y = self.unnormalize(data.output) if self.normalization is not None else data.output
        return (np.repeat(y, 3, axis=-1) * 255).astype(np.uint8)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        raise NotImplementedError(f"[{self}] compute() must be overriden. We inherit it for output_dtype/size etc.")
