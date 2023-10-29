import numpy as np
import pims
import cv2
from overrides import overrides
from ..representation import Representation, RepresentationOutput


def to_categorical(data: np.ndarray, num_classes: int = None) -> np.ndarray:
    """converts the data to categorical. If num classes is not provided, it is infered from the data"""
    data = np.array(data)
    assert data.dtype in (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)
    if num_classes is None:
        num_classes = data.max()
    y = np.eye(num_classes)[data.reshape(-1)].astype(np.uint8)
    # Some bugs for (1, 1) shapes return (1, ) instead of (1, NC)
    MB = data.shape[0]
    y = np.squeeze(y)
    if MB == 1:
        y = np.expand_dims(y, axis=0)
    y = y.reshape(*data.shape, num_classes)
    return y


class KMeans(Representation):
    def __init__(self, video: pims.Video, name: str, dependencies: list, nClusters: int, epsilon: float,
                 maxIterations: int, attempts: int):
        super().__init__(video, name, dependencies)
        self.nClusters = nClusters
        self.epsilon = epsilon
        self.maxIterations = maxIterations
        self.attempts = attempts

    @overrides
    def make(self, t: int) -> RepresentationOutput:
        frame = self.video[t]
        Z = np.float32(frame).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.maxIterations, self.epsilon)
        initialization = cv2.KMEANS_PP_CENTERS
        compactness, labels, centers = cv2.kmeans(Z, self.nClusters, None, criteria, self.attempts, initialization)
        labels = to_categorical(labels[:, 0], self.nClusters)
        data = labels.reshape(frame.shape[0], frame.shape[1], self.nClusters)
        res = {"data": data, "extra": {"centers": centers}}
        return res

    @overrides
    def make_image(self, x: RepresentationOutput) -> np.ndarray:
        res = np.zeros((self.video.frame_shape[0], self.video.frame_shape[1], 3), dtype=np.uint8)
        centers = x["extra"]["centers"]
        x = np.argmax(x["data"], axis=-1)
        for i in range(self.nClusters):
            Where = np.where(x == i)
            res[Where] = centers[i]
        return res
