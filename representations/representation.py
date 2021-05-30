from __future__ import annotations
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Tuple
from functools import lru_cache
from media_processing_lib.image import imgResize
from media_processing_lib.video import MPLVideo
from nwdata.utils import fullPath

# @brief Generic video/image representation
class Representation(ABC):
    def __init__(self, baseDir:Path, name:str, dependencies:Dict[str, Representation], \
        video:MPLVideo, outShape:Tuple[int, int]):
        self.baseDir = baseDir
        self.name = name
        self.video = video
        self.dependencies = dependencies
        self.outShape = outShape

    # @brief Main method of the project. Calls the algorithm's internal logic to transform the current RGB frame into
    # a [0-1] float32 representation.
    @abstractmethod
    def make(self, t:int) -> np.ndarray:
        pass

    # @brief Helper function used to create a plottable [0-255] uint8 representation from a transformed [0-1] float32
    #  representation.
    @abstractmethod
    def makeImage(self, x:np.ndarray) -> np.ndarray:
        pass

    # @brief Method that should automate the entire download/instantiate/resolve any issues with a representation.
    #  Since this is called at every __call__, we should be careful to not instantiate objects for every frame.
    @abstractmethod
    def setup(self):
        pass

    @lru_cache(maxsize=32)
    def __getitem__(self, t:int) -> np.ndarray:
        path = fullPath(self.baseDir / self.name / ("%d.npz" % t))
        if path.exists():
            result = np.load(path)["arr_0"]
        else:
            self.setup()
            rawResult = self.make(t)
            result = imgResize(rawResult, height=self.outShape[0], width=self.outShape[1], onlyUint8=False)
            np.savez_compressed(path, result)

        assert result.shape[0] == self.outShape[0] and result.shape[1] == self.outShape[1], "%s vs %s" % \
            (result.shape, self.outShape)
        assert result.dtype == np.float32 and result.min() >= 0 and result.max() <= 1, \
            "%s: Dtype: %s. Min: %2.2f. Max: %2.2f" % (self, result.dtype, result.min(), result.max())
        return result
