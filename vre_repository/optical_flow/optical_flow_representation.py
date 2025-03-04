"""optical_flow_representation.py -- module implementing an Optical Flow Represenatation generic class"""
from overrides import overrides
import numpy as np
from vre.utils import colorize_optical_flow, ReprOut
from vre.representations import Representation, NpIORepresentation, NormedRepresentationMixin
from vre.vre_video import VREVideo

class OpticalFlowRepresentation(Representation, NpIORepresentation, NormedRepresentationMixin):
    """OpticalFlowRepresentation. Implements flow task-specific stuff."""
    def __init__(self, name: str, delta: int, **kwargs):
        Representation.__init__(self, name, **kwargs)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)
        self.delta = delta

    @property
    @overrides
    def n_channels(self) -> int:
        return 2

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        y = self.unnormalize(data.output) if self.normalization is not None else self.data.output
        return colorize_optical_flow(y)

    def get_delta_frames(self, video: VREVideo, ixs: list[int]) -> np.ndarray:
        """for a given list of frames at .compute() time, return the delta frames required to compute the flow"""
        ixs = list(range(ixs.start, ixs.stop)) if isinstance(ixs, slice) else ixs
        ixs = [min(ix + 1, len(video) - 1) for ix in ixs]
        return video[ixs]
