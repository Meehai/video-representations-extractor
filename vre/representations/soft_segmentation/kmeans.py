import numpy as np
import pims
import cv2
from overrides import overrides
from skimage.color import hsv2rgb
from ...representation import Representation, RepresentationOutput


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

def _generate_diverse_colors(n: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(n):
        hue = i / n  # Vary the hue component
        saturation = 0.7  # You can adjust this value
        value = 0.9  # You can adjust this value
        rgb = hsv2rgb([hue, saturation, value])
        # Convert to 8-bit RGB values (0-255)
        rgb = tuple(int(255 * x) for x in rgb)
        colors.append(rgb)
    return colors

def _get_closest(centers: np.ndarray, colors: np.ndarray) -> np.ndarray:
    M = np.zeros((centers.shape[0], colors.shape[0]))
    for i in range(centers.shape[0]):
        M[i] = np.linalg.norm(centers[i] - colors, axis=1)
    res = [-1] * centers.shape[0]
    for i in range(centers.shape[0]):
        center_ix, color_ix = np.unravel_index(M.argmin(), M.shape)
        res[center_ix] = color_ix
        M[center_ix] = np.inf
        M[:, color_ix] = np.inf
    assert sorted(res) == list(range(centers.shape[0])), res
    return res

class KMeans(Representation):
    def __init__(self, video: pims.Video, name: str, dependencies: list, n_clusters: int, epsilon: float,
                 max_iterations: int, attempts: int):
        super().__init__(video, name, dependencies)
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.attempts = attempts
        self.colors = _generate_diverse_colors(n_clusters)

    def _make_one_frame(self, frame: np.ndarray) -> (np.ndarray, list[tuple[int, int]]):
        Z = np.float32(frame).reshape(-1, 3).copy()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iterations, self.epsilon)
        # initialization = cv2.KMEANS_USE_INITIAL_LABELS
        # centers = np.random.randint(0, 256, (self.n_clusters, 3)).astype(float)
        centers = None
        initialization = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(data=Z, K=self.n_clusters, bestLabels=None, criteria=criteria,
                                                    attempts=self.attempts, flags=initialization, centers=centers)
        labels = to_categorical(labels[:, 0], self.n_clusters)
        data = labels.reshape(frame.shape[0], frame.shape[1], self.n_clusters)
        return data, centers

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        frames = np.array(self.video[t])
        res = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2], self.n_clusters), dtype=np.uint8)
        centers = []
        for i in range(frames.shape[0]):
            np.random.seed(t.start + i)
            one_frame = self._make_one_frame(frames[i])
            res[i] = one_frame[0]
            centers.append({"frame": t.start + i, "centers": one_frame[1]})
        return res, centers

    def _make_one_image(self, x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        res = np.zeros((self.video.frame_shape[0], self.video.frame_shape[1], 3), dtype=np.uint8)
        x = np.argmax(x, axis=-1)
        # ixs = np.argsort(centers, axis=0)[:, 0]
        ixs = _get_closest(centers, np.array(self.colors))
        for i in range(self.n_clusters):
            Where = np.where(x == i)
            res[Where] = self.colors[ixs[i]]
        return res

    @overrides
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        assert extra is not None
        assert len(extra) == len(x), (len(extra), len(x))
        imgs = []
        for _x, _extra in zip(x, extra):
            imgs.append(self._make_one_image(_x, extra["centers"]))
        return np.array(imgs)
