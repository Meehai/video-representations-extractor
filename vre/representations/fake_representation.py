"""FakeRepresentation module"""
from __future__ import annotations
import numpy as np
from overrides import overrides

from vre.utils import image_resize_batch
from .representation import Representation, ReprOut
from .compute_representation_mixin import ComputeRepresentationMixin

class FakeRepresentation(Representation, ComputeRepresentationMixin):
    """FakeRepresentation that is used in unit tests and some basic classes, like RGB."""
    def __init__(self, *args, **kwargs):
        Representation.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        assert len(self.dependencies) == 0, self.dependencies

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, ReprOut] | None = None) -> ReprOut:
        return ReprOut(output=frames)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: ReprOut) -> np.ndarray:
        return repr_data.output

    @overrides
    def size(self, repr_data: ReprOut) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: ReprOut, new_size: tuple[int, int]) -> ReprOut:
        return ReprOut(output=image_resize_batch(repr_data.output, height=new_size[0], width=new_size[1]))
