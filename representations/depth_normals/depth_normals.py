from pathlib import Path
from typing import List, Tuple
from media_processing_lib.video import MPLVideo
import numpy as np

from .cam import fov_diag_to_intrinsic
from representations.representation import Representation
from .utils import get_sampling_grid, get_normalized_coords, depth_to_normals


class DepthNormals(Representation):
    def __init__(self, baseDir:Path, name:str, dependencies:List, video:MPLVideo, outShape:Tuple[int, int],
                 fov:int, windowSize:int, maxDistance:float=None, minValidCount:int=None):
        super().__init__(baseDir, name, dependencies, video, outShape)
        self.fov = fov
        assert windowSize % 2 == 1, "Expected odd window size!"
        self.window_size = windowSize
        self.max_dist = maxDistance if maxDistance is not None else -1
        self.min_valid = minValidCount if minValidCount is not None else 0

        self.sampling_grid = None
        self.normalized_grid = None
        self.K = None

        assert len(dependencies) == 1, "Expected one depth method!"
        self.depth = dependencies[list(dependencies.keys())[0]]

    def make(self, t:int):
        depth = self.depth[t]["rawData"]
        depthHeight, depthWidth = depth.shape[0:2]

        if self.sampling_grid is None:
            self.sampling_grid = get_sampling_grid(depthWidth, depthHeight, self.window_size)

        if self.K is None:
            self.K = fov_diag_to_intrinsic(self.fov, (3840, 2160), (depthWidth, depthHeight))

        if self.normalized_grid is None:
            self.normalized_grid = get_normalized_coords(depthWidth, depthHeight, self.K)

        normals = depth_to_normals(depth, self.sampling_grid, self.normalized_grid, self.max_dist, self.min_valid)
        normals = (normals.astype(np.float32) + 1) / 2
        return normals

    def makeImage(self, x):
        return (x['data'] * 255).astype(np.uint8)

    def setup(self):
        pass