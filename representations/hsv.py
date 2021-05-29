import numpy as np
from skimage.color import rgb2hsv
from media_processing_lib.video import MPLVideo
from typing import Dict
from .representation import Representation

class HSV(Representation):
    def make(self, t:int):
        return np.float32(rgb2hsv(self.video[t]))
         
    def makeImage(self, x):
        return np.uint8(x * 255)

    def setup(self):
        pass