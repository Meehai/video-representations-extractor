from pathlib import Path
from typing import Tuple, Union, List
from media_processing_lib.video import MPLVideo
import numpy as np

from .cam import fov_diag_to_intrinsic
from representations.representation import Representation
from .utils import get_sampling_grid, get_normalized_coords, depth_to_normals

# General method for estimating normals from a depth map (+ intrinsics): a 2D window centered on each pixel is
#  projected into 3D and then a plane is fitted on the 3D pointcloud using SVD.
class DepthNormalsSVD(Representation):
    def __init__(self, name:str, dependencies:List[Union[str, Representation]], dependencyAliases:List[str], \
        fov:int, windowSize:int, maxDistance:float=None, minValidCount:int=None):
        super().__init__(name, dependencies, dependencyAliases)
        self.fov = fov
        assert windowSize % 2 == 1, "Expected odd window size!"
        self.window_size = windowSize
        self.max_dist = maxDistance if maxDistance is not None else -1
        self.min_valid = minValidCount if minValidCount is not None else 0

        assert len(dependencies) == 1, "Expected one depth method!"
        self.depth = dependencies[list(dependencies.keys())[0]]

        self.sampling_grid = None
        self.K = None
        self.normalized_grid = None

    def make(self, t:int):
        depth = self.depth[t]["rawData"]
        normals = depth_to_normals(depth, self.sampling_grid, self.normalized_grid, self.max_dist, self.min_valid)
        normals = (normals.astype(np.float32) + 1) / 2
        return normals

    def makeImage(self, x):
        return (x['data'] * 255).astype(np.uint8)

    def setup(self):
        if not self.sampling_grid is None:
            return
        depthHeight, depthWidth = self.depth[0]["rawData"].shape[:2]
        self.sampling_grid = get_sampling_grid(depthWidth, depthHeight, self.window_size)
        self.K = fov_diag_to_intrinsic(self.fov, (3840, 2160), (depthWidth, depthHeight))
        self.normalized_grid = get_normalized_coords(depthWidth, depthHeight, self.K)
