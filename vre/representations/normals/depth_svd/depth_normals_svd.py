"""Depth normals representation using SVD."""
from overrides import overrides
import numpy as np

from .depth_svd_impl.cam import fov_diag_to_intrinsic
from .depth_svd_impl.utils import get_sampling_grid, get_normalized_coords, depth_to_normals
from ....representation import Representation, RepresentationOutput
from ....utils import image_resize_batch, VREVideo

class DepthNormalsSVD(Representation):
    """
    General method for estimating normals from a depth map (+ intrinsics): a 2D window centered on each pixel is
    projected into 3D and then a plane is fitted on the 3D pointcloud using SVD.
    """
    def __init__(self, sensor_fov: int, sensor_width: int, sensor_height: int, window_size: int,
                 input_downsample_step: int = None, stride: int = None, max_distance: float = None,
                 min_valid_count: int = None, **kwargs):
        assert window_size % 2 == 1, f"Expected odd window size. Got: {window_size}"
        self.depth = None
        self.sensor_fov = sensor_fov
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.window_size = window_size
        self.stride = stride if stride is not None else 1
        self.input_downsample_step = input_downsample_step if input_downsample_step is not None else 1
        self.max_dist = max_distance if max_distance is not None else -1
        self.min_valid = min_valid_count if min_valid_count is not None else 0
        super().__init__(**kwargs)
        assert len(self.dependencies) == 1, "Expected one depth method!"
        self._grid_cache = {}

    @overrides
    def vre_dep_data(self, video: VREVideo, ix: slice) -> dict[str, RepresentationOutput]:
        return {"depths": self.dependencies[0].vre_make(video, ix, make_images=False)[0][0]}

    # pylint: disable=arguments-differ
    @overrides(check_signature=False)
    def make(self, frames: np.ndarray, depths: np.ndarray) -> RepresentationOutput:
        assert len(depths.shape) == 3, f"Expected (T, H, W) got: {depths.shape}"
        res = np.array([self._make_one_depth(depth) for depth in depths])
        return res

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        x_rsz = image_resize_batch(repr_data, height=frames.shape[1], width=frames.shape[2])
        return (x_rsz * 255).astype(np.uint8)

    def _make_one_depth(self, depth: np.ndarray) -> np.ndarray:
        # TODO: batch vectorize this if possible
        sampling_grid, normalized_grid = self._get_grid(depth)
        if self.input_downsample_step is not None:
            depth = depth[:: self.input_downsample_step, :: self.input_downsample_step]
        normals = depth_to_normals(depth, sampling_grid, normalized_grid, self.max_dist, self.min_valid)
        normals = (normals.astype(np.float32) + 1) / 2
        return normals

    def _get_grid(self, depth: np.ndarray) -> (np.ndarray, np.ndarray):
        height, width = depth.shape[:2]
        if (height, width) in self._grid_cache:
            return self._grid_cache[(height, width)]
        if self.input_downsample_step is not None:
            depth = depth[:: self.input_downsample_step, :: self.input_downsample_step]
        depth_height, depth_width = depth.shape[:2]
        sampling_grid = get_sampling_grid(depth_width, depth_height, self.window_size, self.stride)
        K = fov_diag_to_intrinsic(self.sensor_fov, (self.sensor_width, self.sensor_height), (depth_width, depth_height))
        normalized_grid = get_normalized_coords(depth_width, depth_height, K)
        self._grid_cache[(height, width)] = sampling_grid, normalized_grid
        return sampling_grid, normalized_grid
