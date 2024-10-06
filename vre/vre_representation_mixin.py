"""Helper mixin class that adds the VRE relevant methods & properties such that a representation works in vre loop"""
import torch as tr
import numpy as np
from .utils import VREVideo, RepresentationOutput
from .vre_runtime_args import VRERuntimeArgs
from .logger import vre_logger as logger

class VRERepresentationMixin:
    """VRERepresentationMixin class"""
    def __init__(self):
        self.vre_parameters = {}
        self.batch_size: int | None = None
        self.output_size: tuple[int, int] | str | None = None
        self.device: str | tr.device = "cpu"
        self.video: VREVideo | None = None

    # pylint: disable=unused-argument
    def vre_setup(self, **kwargs):
        """
        Setup method for this representation. This is required to run this representation from within VRE.
        We do this setup separately, so we can instatiate the representation without doing any VRE specific setup,
        like loading weights.
        Parameters:
        - video The video we are working with during the VRE run
        - kwargs Any other parameters that can be passed via `build_representation`
        """
        logger.debug(f"[{self}] No runtime setup provided.")

    # pylint: disable=unused-argument
    def vre_dep_data(self, video: VREVideo, ix: slice) -> dict[str, RepresentationOutput]:
        """method used to retrieve the dependencies' data for this frames during a vre run"""
        return {}

    def to(self, device: str | tr.device):
        """
        Support for representation.to(device). Must be updated by all the representations
        that support devices (i.e. cuda torch models)
        """
        self.device = device

    def make_one_frame(self, ix: slice, runtime_args: VRERuntimeArgs) -> tuple[RepresentationOutput, np.ndarray | None]:
        """
        Method used to integrate with VRE. Gets the entire data (video) and a slice of it (ix) and returns the
        representation for that slice. Additionally, if makes_images is set to True, it also returns the image
        representations of this slice.
        """
        assert self.video is not None, f"[{self}] self.video must be set before calling make_one_frame()"
        if tr.cuda.is_available():
            tr.cuda.empty_cache()
        frames = np.array(self.video[ix])
        dep_data = self.vre_dep_data(self.video, ix)
        res_native = self.make(frames, **dep_data)
        if (o_s := runtime_args.output_sizes[self.name]) == "native":
            res = res_native
        elif o_s == "video_shape":
            res = self.resize(res_native, self.video.frame_shape[0:2])
        else:
            res = self.resize(res_native, o_s)
        if isinstance(res, tuple):
            repr_data, extra = res
        else:
            repr_data, extra = res, {}
        imgs = None
        if runtime_args.export_png:
            imgs = self.make_images(frames, res)
        return (repr_data, extra), imgs
