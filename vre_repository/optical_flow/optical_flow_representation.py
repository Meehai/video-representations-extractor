"""optical_flow_representationpy -- module implementing an Optical Flow Represenatation generic class"""
from overrides import overrides
import numpy as np
from vre.utils import colorize_optical_flow, ReprOut
from vre.representations import Representation, NpIORepresentation, NormedRepresentationMixin

class OpticalFlowRepresentation(Representation, NpIORepresentation, NormedRepresentationMixin):
    """OpticalFlowRepresentation. Implements flow task-specific stuff."""
    def __init__(self, name: str, **kwargs):
        Representation.__init__(self, name, **kwargs)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)

    @property
    @overrides
    def n_channels(self) -> int:
        return 2

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        y = self.unnormalize(data.output) if self.normalization is not None else self.data.output
        return colorize_optical_flow(y)
