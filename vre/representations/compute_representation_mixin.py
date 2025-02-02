"""ComputeRepresentationMixin module"""
from abc import ABC, abstractmethod

from ..vre_video import VREVideo
from ..logger import vre_logger as logger

class ComputeRepresentationMixin(ABC):
    """ComputeRepresentationMixin for representations that can be computed"""
    def __init__(self):
        self._batch_size: int | None = None

    @abstractmethod
    def compute(self, video: VREVideo, ixs: list[int]):
        """given input data (batch of images/frewes), compute the output data of this representation"""

    @property
    def batch_size(self) -> int:
        """the batch size that is used during the VRE run for computation"""
        if self._batch_size is None:
            logger.warning(f"[{self}] No batch_size set, returning 1. Call set_compute_params")
            return 1
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs: int):
        assert isinstance(bs, int) and bs >= 1, type(bs)
        self._batch_size = bs

    def set_compute_params(self, **kwargs):
        """set the compute parameters for the representation"""
        attributes = ["batch_size"]
        res = ""
        assert set(kwargs).issubset(attributes), (list(kwargs), attributes)
        for attr in attributes:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
                res += f"\n-{attr}: {kwargs[attr]}"
        if len(res) > 0:
            logger.debug(f"[{self}] Set node specific 'Compute' params: {res}")
