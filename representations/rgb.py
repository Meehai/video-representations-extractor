import numpy as np
from .representation import Representation

class RGB(Representation):
    def make(self, frame):
        return np.float32(frame) / 255
    
    def makeImage(self, x):
        return np.uint8(x * 255)

    def setup(self):
        pass