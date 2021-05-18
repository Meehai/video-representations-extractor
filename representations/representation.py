import numpy as np
from abc import ABC, abstractmethod

# @brief Generic video/image representation
class Representation(ABC):
    @abstractmethod
    def makeFrame(self, frame:np.ndarray):
        pass

    def __call__(self, frame:np.ndarray) -> np.ndarray:
        assert len(frame.shape) == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8
        result = self.makeFrame(frame)
        assert result.shape[0] == frame.shape[0] and result.shape[1] == frame.shape[1] \
            and result.dtype == np.float32 and result.min() >= 0 and result.max() <= 1
        return result