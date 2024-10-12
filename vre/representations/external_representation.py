"""ExternalRepresentation module"""
from __future__ import annotations
from typing import Callable
import numpy as np
from overrides import overrides

from vre.utils import image_resize_batch
from .representation import Representation, ReprOut

class ExternalRepresentation(Representation):
    """
    A placeholder for external data that we cannot compute (i.e., sfm, edges_gb etc.). Must provide a plot fn and
    optionally a resize_fn.
    """
    def __init__(self, name: str, make_image_fn: Callable,
                 resize_fn: Callable[[ReprOut, tuple[int, int]], ReprOut] | None = None):
        super().__init__(name=name, dependencies=[])
        assert isinstance(make_image_fn, Callable), type(make_image_fn)
        assert isinstance(resize_fn, (Callable, type(None))), type(resize_fn)
        self.make_image_fn = make_image_fn
        self.resize_fn = resize_fn

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, ReprOut] | None = None) -> ReprOut:
        raise ValueError(f"[{self}] ExternalRepresentation.make() cannot be called. Data must be loaded externally")

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: ReprOut) -> np.ndarray:
        return self.make_image_fn(frames, repr_data)

    @overrides
    def size(self, repr_data: ReprOut) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: ReprOut, new_size: tuple[int, int]) -> ReprOut:
        if self.resize_fn is None:
            return ReprOut(output=image_resize_batch(repr_data.output, height=new_size[0], width=new_size[1]))
        return self.resize_fn(repr_data, new_size)
