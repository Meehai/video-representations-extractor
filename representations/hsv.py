import numpy as np
from skimage.color import rgb2hsv
from .representation import Representation

class HSV(Representation):
    def make(self, video, t):
        return np.float32(rgb2hsv(video[t]))
         
    def makeImage(self, x):
        return np.uint8(x * 255)

    def setup(self):
        pass