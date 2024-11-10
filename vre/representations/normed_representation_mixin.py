"""NormedRepresentation module"""
import numpy as np

class NormedRepresentationMixin:
    """NormedRepresentation implementation"""
    def __init__(self):
        self.normalization: str | None = None
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def set_normalization(self, normalization: str, stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """sets the normalization"""
        assert normalization in ("min_max", "standardization"), normalization
        assert isinstance(stats, tuple) and len(stats) == 4, stats
        self.normalization = normalization
        self.min, self.max, self.mean, self.std = [np.array(x) for x in stats]

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """normalizes a data point"""
        assert self.stats is not None, "stats must be called before task.normalize(x) via task.set_normalization()"
        if self.normalization == "min_max":
            return self.min_max(x)
        return self.standardize(x)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        """Given a normalized representation using either self.normalize or self.standardize, turn it back"""
        assert self.stats is not None, "stats must be called before task.unnormalize(x) via task.set_normalization()"
        if self.normalization == "min_max":
            return x * (self.max - self.min) + self.min
        return x * self.std + self.mean

    def min_max(self, x: np.ndarray) -> np.ndarray:
        """normalizes a data point read with self.load_from_disk(path) using external min/max information"""
        assert self.stats is not None, "stats must be called before task.min_max(x) via task.set_normalization()"
        res = np.nan_to_num(((x.astype(np.float32) - self.min) / (self.max - self.min)), False, 0, 0, 0)
        return res.astype(np.float32)

    def standardize(self, x: np.ndarray) -> np.ndarray:
        """standardizes a data point read with self.load_from_disk(path) using external min/max information"""
        assert self.stats is not None, "stats must be called before task.staandardize(x) via task.set_normalization()"
        res = np.nan_to_num(((x.astype(np.float32) - self.mean) / self.std), False, 0, 0, 0)
        res[(res < -10) | (res > 10)] = 0
        return res.astype(np.float32)

    @property
    def stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Returns a tuple with the min/max/mean/std tensors, or None if normalization is None"""
        if self.normalization is None:
            return None
        assert self.min is not None, f"Use self.set_normalization first for {self.normalization=}"
        return self.min, self.max, self.mean, self.std
