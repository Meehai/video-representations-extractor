"""Helper mixin class that adds the VRE relevant methods & properties such that a representation works in vre loop"""
import torch as tr
import numpy as np
from .utils import VREVideo, RepresentationOutput

class VRERepresentationMixin:
    """VRERepresentationMixin class"""
    def __init__(self):
        self.batch_size: int | None = None
        self.output_size: tuple[int, int] | str | None = None
        self.device: str | tr.device = "cpu"
        self.video: VREVideo | None = None

    def vre_setup(self):
        """
        Setup method for this representation. This is required to run this representation from within VRE.
        We do this setup separately, so we can instatiate the representation without doing any VRE specific setup,
        like loading weights.
        """
        raise RuntimeError(f"[{self}] No runtime setup provided. Override with a 'pass' method if not needed.")

    def vre_dep_data(self, ix: slice) -> dict[str, RepresentationOutput]:
        """iteratively collects all the dependencies needed by this representation"""
        assert self.video is not None, f"[{self}] self.video must be set before calling this"
        return {dep.name: dep.vre_make(ix) for dep in self.dependencies}

    def vre_make(self, ix: slice) -> RepresentationOutput:
        """wrapper on top of make() that is ran in VRE context. TODO: support loading from disk if needed"""
        assert self.video is not None, f"[{self}] self.video must be set before calling this"
        if tr.cuda.is_available():
            tr.cuda.empty_cache()
        frames = np.array(self.video[ix])
        dep_data = self.vre_dep_data(ix)
        res = self.make(frames, dep_data)
        assert isinstance(res, RepresentationOutput) and not isinstance(res.output, RepresentationOutput), (self, res)
        return res

    def to(self, device: str | tr.device):
        """
        Support for representation.to(device). Must be updated by all the representations
        that support devices (i.e. cuda torch models)
        """
        raise NotImplementedError("TODO")
