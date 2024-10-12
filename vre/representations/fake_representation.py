"""FakeRepresentation module"""
from __future__ import annotations
import numpy as np
from overrides import overrides

from vre.utils import image_resize_batch
from .representation import Representation, ReprOut

class FakeRepresentation(Representation):
    """FakeRepresentation that is used in unit tests. It was also a placeholder for external data (sfm, edgesgb etc.)"""
    def __init__(self, name: str, dependencies: list[str] | None = None):
        super().__init__(name=name, dependencies=[] if dependencies is None else dependencies)

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
