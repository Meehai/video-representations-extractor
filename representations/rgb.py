import numpy as np
from .representation import Representation

class RGB(Representation):
    def make(self, video, t):
        return np.float32(video[t]) / 255
    
    def makeImage(self, x):
        return np.uint8(x * 255)

    def setup(self):
        pass