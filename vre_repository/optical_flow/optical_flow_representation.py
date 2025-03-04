"""optical_flow_representation.py -- module implementing an Optical Flow Represenatation generic class"""
from overrides import overrides
import numpy as np
from vre.utils import colorize_optical_flow, ReprOut, clip
from vre.representations import Representation, NpIORepresentation, NormedRepresentationMixin
from vre.vre_video import VREVideo

def _get_delta_frames(video: VREVideo, ixs: list[int], delta: int) -> list[int]:
    assert delta != 0, delta
    return [clip(ix + delta, 0, len(video) - 1) for ix in ixs]

class OpticalFlowRepresentation(Representation, NpIORepresentation, NormedRepresentationMixin):
    """OpticalFlowRepresentation. Implements flow task-specific stuff."""
    def __init__(self, name: str, delta: int, **kwargs):
        Representation.__init__(self, name, **kwargs)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)
        assert isinstance(delta, int) and delta != 0, delta
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
        return video[_get_delta_frames(video, ixs, self.delta)]
