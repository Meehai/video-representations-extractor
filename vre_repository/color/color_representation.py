
"""color_representation.py -- module implementing an 'Color' (RGB-like) Represenatation generic class"""
from overrides import overrides
import numpy as np
from vre import Representation, ReprOut
from vre.representations.mixins import NpIORepresentationMixin, NormedRepresentationMixin, ResizableRepresentationMixin

class ColorRepresentation(NormedRepresentationMixin, NpIORepresentationMixin,
                          ResizableRepresentationMixin, Representation):
    """ColorRepresentation -- a wrapper over all 3-channeled colored representations"""
    def __init__(self, name: str, **kwargs):
        Representation.__init__(self, name, **kwargs)
        NormedRepresentationMixin.__init__(self)
        NpIORepresentationMixin.__init__(self)
        ResizableRepresentationMixin.__init__(self)

    @property
    @overrides
    def n_channels(self) -> int:
        return 3

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        y = self.unnormalize(data.output) if self.normalization is not None else data.output
        return y.astype(np.uint8)
