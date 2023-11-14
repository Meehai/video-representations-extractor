"""RGB representation"""
import numpy as np
from overrides import overrides
from ..representation import Representation, RepresentationOutput

class RGB(Representation):
    """RGB representation"""
    @overrides
    def make(self, frames: np.ndarray) -> RepresentationOutput:
        return frames.astype(np.float32) / 255

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        return frames
