import numpy as np
import cv2
from pathlib import Path
from media_processing_lib.video import MPLVideo
from typing import Dict, Tuple, List
from matplotlib.cm import gray
from .representation import Representation

class Canny(Representation):
    def __init__(self, baseDir:Path, name:str, dependencies:List, video:MPLVideo, outShape:Tuple[int, int], \
        threshold1:float, threshold2:float, apertureSize:int, L2gradient:bool):
        super().__init__(baseDir, name, dependencies, video, outShape)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.apertureSize = apertureSize
        self.L2gradient = L2gradient

    def make(self, t:int):
        frame = self.video[t]
        res = frame * 0
        res = cv2.Canny(frame, self.threshold1, self.threshold2, res, self.apertureSize, self.L2gradient)
        res = np.float32(res) / 255
        return res
         
    def makeImage(self, x):
        return np.uint8(255 * gray(x["data"])[..., 0 : 3])

    def setup(self):
        pass