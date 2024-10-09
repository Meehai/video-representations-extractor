"""Helper mixin class that adds the VRE relevant methods & properties such that a representation works in vre loop"""
from pathlib import Path
import torch as tr
import numpy as np
from .utils import VREVideo, RepresentationOutput
from .logger import vre_logger as logger

class VRERepresentationMixin:
    """VRERepresentationMixin class"""
    def __init__(self):
        self.batch_size: int | None = None
        self.output_size: tuple[int, int] | str | None = None
        self.device: str | tr.device = "cpu"
        self.video: VREVideo | None = None
        self.output_dir: Path | None = None

    def vre_setup(self):
        """
        Setup method for this representation. This is required to run this representation from within VRE. We do this
        setup separately, so we can instatiate the object without doing any VRE specific setup, like loading weights.
        """
        raise RuntimeError(f"[{self}] No runtime setup provided. Override with a 'pass' method if not needed.")

    def vre_dep_data(self, ix: slice) -> dict[str, RepresentationOutput]:
        """iteratively collects all the dependencies needed by this representation"""
        assert self.video is not None, f"[{self}] self.video must be set before calling this"
        return {dep.name: dep.vre_make(ix) for dep in self.dependencies}

    def vre_make(self, ix: slice) -> RepresentationOutput:
        """wrapper on top of make() that is ran in VRE context. TODO: support loading from disk if needed"""
        assert self.video is not None, f"[{self}] self.video must be set before calling this"
        if str(self.device).startswith("cuda"):
            tr.cuda.empty_cache()
        if self.output_dir is not None:
            npy_paths: list[Path] = [self.output_dir / self.name / f"npy/{i}.npz" for i in range(ix.start, ix.stop)]
            extra_paths: list[Path] = [self.output_dir / self.name / f"npy/{i}_extra.npz"
                                       for i in range(ix.start, ix.stop)]
            if all(x.exists() for x in npy_paths):
                data, extra = np.stack([np.load(x)["arr_0"] for x in npy_paths]), None
                if all(x.exists() for x in extra_paths):
                    extra = [np.load(x, allow_pickle=True)["arr_0"].item() for x in extra_paths]
                logger.debug2(f"[{self}] Slice: [{ix.start}:{ix.stop - 1}]. All data found on disk and loaded")
                return RepresentationOutput(output=data, extra=extra)
        frames = np.array(self.video[ix])
        dep_data = self.vre_dep_data(ix)
        res = self.make(frames, dep_data)
        assert isinstance(res, RepresentationOutput) and not isinstance(res.output, RepresentationOutput), (self, res)
        return res

    def vre_free(self):
        """Needed to deallocate stuff from cuda mostly. After this, you need to run vre_setup() again."""
        if str(self.device).startswith("cuda"):
            logger.warning(f"[{self}] Representation has device that is not CPU {self.device}, with default vre_free()")
