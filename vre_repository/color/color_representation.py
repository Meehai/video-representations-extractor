
"""color_representation.py -- module implementing an 'Color' (RGB-like) Represenatation generic class"""
from overrides import overrides
import numpy as np
from vre.representations import Representation, NpIORepresentation, NormedRepresentationMixin
from vre.utils import ReprOut

class ColorRepresentation(Representation, NormedRepresentationMixin, NpIORepresentation):
    """ColorRepresentation -- a wrapper over all 3-channeled colored representations"""
    def __init__(self, name: str, **kwargs):
        Representation.__init__(self, name, **kwargs)
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
