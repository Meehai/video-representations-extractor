"""NormedRepresentation module"""
import torch as tr

class NormedRepresentation:
    """NormedRepresentation implementation"""
    def __init__(self):
        self.normalization: str | None = None
        self.min: tr.Tensor | None = None
        self.max: tr.Tensor | None = None
        self.mean: tr.Tensor | None = None
        self.std: tr.Tensor | None = None

    def set_normalization(self, normalization: str, stats: tuple[tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor]):
        """sets the normalization"""
        assert normalization in ("min_max", "standardization"), normalization
        assert isinstance(stats, tuple) and len(stats) == 4, stats
        self.normalization = normalization
        self.min, self.max, self.mean, self.std = stats

    def normalize(self, x: tr.Tensor) -> tr.Tensor:
        """normalizes a data point read with self.load_from_disk(path) using external min/max information"""
        assert self.min is not None, "self.statistics must be set from reader before task.normalize(x)"
        return ((x.float() - self.min) / (self.max - self.min)).nan_to_num(0, 0, 0).float()

    def standardize(self, x: tr.Tensor) -> tr.Tensor:
        """standardizes a data point read with self.load_from_disk(path) using external min/max information"""
        assert self.min is not None, "self.statistics must be set from reader before task.normalize(x)"
        res = ((x.float() - self.mean) / self.std).nan_to_num(0, 0, 0)
        res[(res < -10) | (res > 10)] = 0
        return res.float()

    @property
    def stats(self) -> tuple[tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor] | None:
        """Returns a tuple with the min/max/mean/std tensors, or None if normalization is None"""
        if self.normalization is None:
            return None
        assert self.min is not None, f"Use self.set_normalization first for {self.normalization=}"
        return self.min, self.max, self.mean, self.std
