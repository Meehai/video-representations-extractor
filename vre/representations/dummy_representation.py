from __future__ import annotations
import numpy as np
from overrides import overrides
from typing import Callable

from ..representation import Representation

# @brief A representation that is a placeholder for a set of npz files that were precomputted by other (not yet
#  not yet supported) algorithms, such as Sfm, SemanticGB, weird pre-trained networks etc. The only thing it needs is
#  to provide the correct files (0.npz, ..., N.npz) as well as a function to plot the data to human viewable format.
class DummyRepresentation(Representation):
    def __init__(self, name: str, make_imageFn: Callable):
        super().__init__(name, dependencies=[], dependencyAliases=[])
        assert isinstance(make_imageFn, Callable)
        self.make_imageFn = make_imageFn

    @overrides
    def make(self, t: int) -> np.ndarray:
        assert False, f"Dummy representation ({self.name}) has no precomputted npz files!"

    @overrides
    def make_image(self, x: dict) -> np.ndarray:
        return self.make_imageFn(x)
