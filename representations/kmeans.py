import numpy as np
import cv2
from pathlib import Path
from media_processing_lib.video import MPLVideo
from typing import Dict, Tuple, List
from nwdata.utils import toCategorical
from .representation import Representation

class KMeans(Representation):
    def __init__(self, name:str, dependencies:List, saveResults:str, dependencyAliases:List[str], \
        nClusters:int, epsilon:float, maxIterations:int, attempts:int):
        super().__init__(name, dependencies, saveResults, dependencyAliases)
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
        data = labels.reshape(frame.shape[0], frame.shape[1], self.nClusters)
        res = {"data" : data, "extra" : {"centers" : centers}}
        return res
         
    def makeImage(self, x):
        res = np.zeros((self.outShape[0], self.outShape[1], 3), dtype=np.uint8)
        centers = x["extra"]["centers"]
        x = np.argmax(x["data"], axis=-1)
        for i in range(self.nClusters):
            Where = np.where(x == i)
            res[Where] = centers[i]
        return res

    def setup(self):
        pass