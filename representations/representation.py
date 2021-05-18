import numpy as np
from abc import ABC, abstractmethod

# @brief Generic video/image representation
class Representation(ABC):
    # @brief Main method of the project. Calls the algorithm's internal logic to transform the current RGB frame into
    # a [0-1] float32 representation.
    @abstractmethod
    def make(self, frame:np.ndarray):
        pass

    # @brief Helper function used to create a plottable [0-255] uint8 representation from a transformed [0-1] float32
    #  representation.
    @abstractmethod
    def makeImage(self, x:np.ndarray):
        pass

    def __call__(self, frame:np.ndarray) -> np.ndarray:
        assert len(frame.shape) == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8
        result = self.make(frame)
        assert result.shape[0] == frame.shape[0] and result.shape[1] == frame.shape[1] \
            and result.dtype == np.float32 and result.min() >= 0 and result.max() <= 1
        return result