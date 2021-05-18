import numpy as np
from abc import ABC, abstractmethod

# @brief Generic video/image representation
class Representation(ABC):
    Obj = None
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def instantiate(self, *args, **kwargs):
        pass

    @abstractmethod
    def makeFrame(self, frame:np.ndarray):
        pass

    def getObj(self):
        if Representation.Obj == None:
            Representation.Obj = self.instantiate(*self.args, **self.kwargs)
        return Representation.Obj

    def __call__(self, frame:np.ndarray) -> np.ndarray:
        assert len(frame.shape) == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8
        result = self.getObj().makeFrame(frame)
        assert result.shape[0] == frame.shape[0] and result.shape[1] == frame.shape[1] \
            and result.dtype == np.float32 and result.min() >= 0 and result.max() <= 1