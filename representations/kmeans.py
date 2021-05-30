import numpy as np
import cv2
from pathlib import Path
from media_processing_lib.video import MPLVideo
from typing import Dict, Tuple, List
from nwdata.utils import toCategorical
from .representation import Representation

class KMeans(Representation):
    def __init__(self, baseDir:Path, name:str, dependencies:List, video:MPLVideo, outShape:Tuple[int, int], \
        nClusters:int, epsilon:float, maxIterations:int, attempts:int):
        super().__init__(baseDir, name, dependencies, video, outShape)
        self.nClusters = nClusters
        self.epsilon = epsilon
        self.maxIterations = maxIterations
        self.attempts  = attempts

    def make(self, t:int):
        frame = self.video[t]
        Z = np.float32(frame).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.maxIterations, self.epsilon)
        initialization = cv2.KMEANS_PP_CENTERS
        compactness, labels, centers = cv2.kmeans(Z, self.nClusters, None, criteria, self.attempts, initialization)
        labels = toCategorical(labels[:, 0], self.nClusters)
        res = labels.reshape(frame.shape[0], frame.shape[1], self.nClusters)
        self.currentFrame = frame
        return res
         
    def makeImage(self, x):
        currentFrame = self.video[self.t]
        res = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
        x = np.argmax(x, axis=-1)
        for i in range(self.nClusters):
            Where = np.where(x == i)
            colors = np.median(currentFrame[Where], axis=0)
            res[Where] = colors
        return res

    def setup(self):
        pass