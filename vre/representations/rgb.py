"""RGB representation"""
import numpy as np
from overrides import overrides
from ..representation import Representation, RepresentationOutput
from ..utils import image_resize_batch

class RGB(Representation):
    """RGB representation"""
    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, RepresentationOutput] | None = None) -> RepresentationOutput:
        return RepresentationOutput(output=frames)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        return repr_data.output

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return RepresentationOutput(output=image_resize_batch(repr_data.output, height=new_size[0], width=new_size[1]))

    @overrides
    def vre_setup(self):
        pass
