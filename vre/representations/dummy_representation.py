"""DummyRepresentation module"""
from __future__ import annotations
import numpy as np
from overrides import overrides
from typing import Callable

from ..representation import Representation, RepresentationOutput

class DummyRepresentation(Representation):
    """
    A representation that is a placeholder for a set of npz files that were precomputted by other (not yet
    not yet supported) algorithms, such as Sfm, SemanticGB, weird pre-trained networks etc. The only thing it needs is
    to provide the correct files (0.npz, ..., N.npz) as well as a function to plot the data to human viewable format.
    """
    def __init__(self, name: str, make_image_fn: Callable):
        super().__init__(name, dependencies=[], dependencyAliases=[])
        assert isinstance(make_image_fn, Callable)
        self.make_image_fn = make_image_fn

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        assert False, f"Dummy representation ({self.name}) has no precomputted npz files!"

    @overrides
    def make_images(self, t: slice, x: np.ndarray, extra: dict | None) -> np.ndarray:
        return self.make_image_fn(x)
