"""normals_representation.py -- module implementing an Normals & Cameras Normals Represenatation generic class"""
from overrides import overrides
import numpy as np
from vre.utils import ReprOut
from vre.representations import Representation, NpIORepresentation, NormedRepresentationMixin

class NormalsRepresentation(Representation, NpIORepresentation, NormedRepresentationMixin):
    """NormalsRepresentation -- CV representation for world and camera normals"""
    def __init__(self, name: str, **kwargs):
        Representation.__init__(self, name, **kwargs)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        y = self.unnormalize(data.output) if self.normalization is not None else self.data.output
        return (y * 255).astype(np.uint8)

    @property
    @overrides
    def n_channels(self) -> int:
        return 3
