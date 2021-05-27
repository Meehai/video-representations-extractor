import numpy as np
from abc import ABC, abstractmethod
from media_processing_lib.video import MPLVideo

# @brief Generic video/image representation
class Representation(ABC):
    # @brief Main method of the project. Calls the algorithm's internal logic to transform the current RGB frame into
    # a [0-1] float32 representation.
    @abstractmethod
    def make(self, video:MPLVideo, t:int):
        pass

    # @brief Helper function used to create a plottable [0-255] uint8 representation from a transformed [0-1] float32
    #  representation.
    @abstractmethod
    def makeImage(self, x:np.ndarray):
        pass

    # @brief Method that should automate the entire download/instantiate/resolve any issues with a representation
    @abstractmethod
    def setup(self):
        pass

    def __call__(self, video:MPLVideo, t:int) -> np.ndarray:
        result = self.make(video, t)
        # assert result.dtype == np.float32 and result.min() >= 0 and result.max() <= 1, \
        #     "%s: Dtype: %s. Min: %2.2f. Max: %2.2f" % (self, result.dtype, result.min(), result.max())
        return result