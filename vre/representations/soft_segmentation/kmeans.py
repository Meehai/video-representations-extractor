"""KMeans representation"""
import numpy as np
from overrides import overrides
from ...representation import Representation, RepresentationOutput
from ...utils import to_categorical, generate_diverse_colors, image_resize_batch

def _get_closest(centers: np.ndarray, colors: np.ndarray) -> list:
    # pylint: disable=unbalanced-tuple-unpacking
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
    """KMeans representation"""
    def __init__(self, n_clusters: int, epsilon: float, max_iterations: int, attempts: int, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.attempts = attempts
        self.colors = generate_diverse_colors(n_clusters, saturation=0.7, value=0.9)

    @overrides
    def make(self, frames: np.ndarray) -> RepresentationOutput:
        res = np.zeros((*frames.shape[0:3], self.n_clusters), dtype=np.uint8)
        centers = []
        for i in range(frames.shape[0]):
            one_frame = self._make_one_frame(frames[i])
            res[i] = one_frame[0]
            centers.append({"centers": one_frame[1]})
        return res, centers

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        x, extra = repr_data
        assert extra is not None
        assert len(extra) == len(x), (len(extra), len(x))
        imgs = []
        for _x, _extra in zip(x, extra):
            imgs.append(self._make_one_image(_x, _extra["centers"]))
        return np.array(imgs)

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        assert isinstance(repr_data, tuple), type(repr_data) # as make() returned it
        return repr_data[0].shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        assert isinstance(repr_data, tuple), type(repr_data)
        data_resized = image_resize_batch(repr_data[0], *new_size, interpolation="nearest")
        return (data_resized, repr_data[1]) # no need to resize centers as they are in color space

    def _make_one_frame(self, frame: np.ndarray) -> (np.ndarray, list[tuple[int, int]]):
        # pylint: disable=import-outside-toplevel, too-many-function-args
        import cv2
        Z = np.float32(frame).reshape(-1, 3).copy()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iterations, self.epsilon)
        # TODO: this simply doesn't work, cv2 bugs out. We should replace this implementation with a better one.
        # initialization = cv2.KMEANS_USE_INITIAL_LABELS
        # centers = np.random.randint(0, 256, (self.n_clusters, 3)).astype(float)
        centers = None
        initialization = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centers = cv2.kmeans(data=Z, K=self.n_clusters, bestLabels=None, criteria=criteria,
                                        attempts=self.attempts, flags=initialization, centers=centers)
        labels = to_categorical(labels[:, 0], self.n_clusters)
        data = labels.reshape(frame.shape[0], frame.shape[1], self.n_clusters)
        return data, centers

    def _make_one_image(self, x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        res = np.zeros((*x.shape[0:2], 3), dtype=np.uint8)
        x = np.argmax(x, axis=-1)
        ixs = _get_closest(centers, np.array(self.colors))
        for i in range(self.n_clusters):
            where = np.where(x == i)
            res[where] = self.colors[ixs[i]]
        return res
