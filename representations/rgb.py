import numpy as np
from media_processing_lib.video import MPLVideo
from typing import Dict
from .representation import Representation

class RGB(Representation):
    def make(self, video:MPLVideo, t:int, depenedencyInputs:Dict[str, np.ndarray]) -> np.ndarray:
        return np.float32(video[t]) / 255
    
    def makeImage(self, x:np.ndarray) -> np.ndarray:
        return np.uint8(x * 255)

    def setup(self):
        pass