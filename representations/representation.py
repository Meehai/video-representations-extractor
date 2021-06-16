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
    def __init__(self, name:str, dependencies:Dict[str, Representation]):
        self.name = name
        self.dependencies = dependencies
        self.video = None
        self.baseDir = None
        self.outShape = None

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

    def setVideo(self, video:MPLVideo):
        self.video = video

    def setBaseDir(self, baseDir:path):
        self.baseDir = baseDir
    
    def setOutShape(self, outShape:Tuple[int, int]):
        self.outShape = outShape

    # @brief The representation[t] method, which returns the data of this representation. The format is a dict:
    #  - data: a raw np.ndarray of shape (outShape[0], outShape[1], C)
    #  - rawData: a raw np.ndarray of whatever shape the representation decided to spit out for current frame
    #  - extra: whatever extra stuff this representation needs (i.e. cluster centroids, perhaps useful for t+1)
    # @param[in] The time t at which representation[t] = repr(video[t])
    # @return The representation at time t
    @lru_cache(maxsize=32)
    def __getitem__(self, t:int) -> Dict[str, np.ndarray]:
        assert not self.video is None, "Call setVideo first"
        assert not self.baseDir is None, "Call setBaseDir first"
        assert not self.outShape is None, "Call setOutShape first"
        t = t % len(self.video)
        path = fullPath(self.baseDir / self.name / ("%d.npz" % t))
        assert t >= 0 and t < len(self.video)

        # Try to load from the disk first, so we avoid computting representation[t] multiple times
        try:
            result = np.load(path, allow_pickle=True)["arr_0"]
            # Support for legacy storing only the array, not raw results.
            if result.dtype != np.object:
                result = {"data":result, "rawData":result, "extra":{}}
            else:
                result = result.item()
        except Exception:
            # Call the setup method so, for example, we can download weights of pretrained networks if needed
            self.setup()
            # Get the raw result of this representation
            rawResult = self.make(t)
            if isinstance(rawResult, np.ndarray):
                rawResult = {"data" : rawResult, "extra" : {}}
            rawData, extra = rawResult["data"], rawResult["extra"]
            # Based on the raw result, resize it to outputShape
            interpolation = "nearest" if rawData.dtype == np.uint8 else "bilinear"
            resizedData = imgResize(rawData, height=self.outShape[0], width=self.outShape[1], \
                onlyUint8=False, interpolation=interpolation)

            result = {
                "data" : resizedData,
                "rawData" : rawData,
                "extra" : extra
            }
            # Store it to disk.
            np.savez_compressed(path, result)

        assert isinstance(result, dict) and "data" in result and "rawData" in result and "extra" in result, \
            "Representation: %s. Type: %s. Keys: %s" % (self, type(result), result.keys())
        data = result["data"]
        assert data.shape[0] == self.outShape[0] and data.shape[1] == self.outShape[1], \
            "%s vs %s" % (data.shape, self.outShape)
        assert data.dtype in (np.float32, np.uint8) and data.min() >= 0 and data.max() <= 1, \
            "%s: Dtype: %s. Min: %2.2f. Max: %2.2f" % (self.name, data.dtype, data.min(), data.max())
        return result
