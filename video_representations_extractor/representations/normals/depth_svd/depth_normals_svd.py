from typing import Union, List
import numpy as np

from .cam import fov_diag_to_intrinsic
from .utils import get_sampling_grid, get_normalized_coords, depth_to_normals
from ...representation import Representation

# General method for estimating normals from a depth map (+ intrinsics): a 2D window centered on each pixel is
#  projected into 3D and then a plane is fitted on the 3D pointcloud using SVD.
class DepthNormalsSVD(Representation):
    def __init__(self, name:str, dependencies:List[Union[str, Representation]], saveResults:str, \
                 dependencyAliases:List[str], fov:int, sensorWidth:int, sensorHeight:int,
                 windowSize:int, inputDownsampleStep:int=None, stride:int=None,
                 maxDistance:float=None, minValidCount:int=None):
        assert len(dependencies) == 1, "Expected one depth method!"
        super().__init__(name, dependencies, saveResults, dependencyAliases)
        self.fov = fov
        self.sensorWidth = sensorWidth
        self.sensorHeight = sensorHeight
        assert windowSize % 2 == 1, "Expected odd window size!"
        self.window_size = windowSize
        self.stride = stride if stride is not None else 1
        self.inputDownsampleStep = inputDownsampleStep if inputDownsampleStep is not None else 1
        self.max_dist = maxDistance if maxDistance is not None else -1
        self.min_valid = minValidCount if minValidCount is not None else 0

        self.depth = None
        self.sampling_grid = None
        self.K = None
        self.normalized_grid = None

    def make(self, t:int):
        depth = self.depth[t]["data"]
        if self.inputDownsampleStep is not None:
            depth = depth[::self.inputDownsampleStep, ::self.inputDownsampleStep]
        normals = depth_to_normals(depth, self.sampling_grid, self.normalized_grid, self.max_dist, self.min_valid)
        normals = (normals.astype(np.float32) + 1) / 2
        return normals

    def makeImage(self, x):
        return (x['data'] * 255).astype(np.uint8)

    def setup(self):
        if not self.sampling_grid is None:
            return
        self.depth = self.dependencies[0]
        depth = self.depth[0]["data"]
        if self.inputDownsampleStep is not None:
            depth = depth[::self.inputDownsampleStep, ::self.inputDownsampleStep]
        depthHeight, depthWidth = depth.shape[:2]
        self.sampling_grid = get_sampling_grid(depthWidth, depthHeight, self.window_size, self.stride)
        self.K = fov_diag_to_intrinsic(self.fov, (self.sensorWidth, self.sensorHeight), (depthWidth, depthHeight))
        self.normalized_grid = get_normalized_coords(depthWidth, depthHeight, self.K)
