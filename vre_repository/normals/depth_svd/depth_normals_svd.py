"""Depth normals representation using SVD."""
from overrides import overrides
import numpy as np

from vre.utils import MemoryData, ReprOut
from vre.vre_video import VREVideo
from vre.representations import ComputeRepresentationMixin
from vre_repository.normals import NormalsRepresentation

from .depth_svd_impl import depths_to_normals

class DepthNormalsSVD(NormalsRepresentation, ComputeRepresentationMixin):
    """
    General method for estimating normals from a depth map (+ intrinsics): a 2D window centered on each pixel is
    projected into 3D and then a plane is fitted on the 3D pointcloud using SVD.
    """
    def __init__(self, sensor_fov: int, sensor_size: tuple[int, int], window_size: int,
                 input_downsample_step: int = None, stride: int = None, **kwargs):
        NormalsRepresentation.__init__(self, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        assert window_size % 2 == 1, f"Expected odd window size. Got: {window_size}"
        self.sensor_fov = sensor_fov
        self.sensor_size = sensor_size
        self.window_size = window_size
        self.stride = stride or 1
        self.input_downsample_step = input_downsample_step or 1
        assert len(self.dependencies) == 1, f"Expected exactly one depth method, got: {self.dependencies}"

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        assert (A := self.dependencies[0].data) is not None and self.dependencies[0].data.key == ixs, (A.key, ixs)
        depths = self.dependencies[0].data.output
        assert len(depths.shape) == 4 and depths.shape[-1] == 1, f"Expected (B, H, W, 1) got: {depths.shape}"
        res = depths_to_normals(depths[..., 0], self.sensor_fov, self.window_size, self.stride,
                                self.sensor_size, self.input_downsample_step)
        res = (MemoryData(res).astype(np.float32) + 1) / 2
        self.data = ReprOut(frames=video[ixs], output=res, key=ixs)
