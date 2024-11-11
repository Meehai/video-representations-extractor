"""ComputeRepresentationMixin module"""
from abc import ABC, abstractmethod
import numpy as np

from ..utils import VREVideo
from ..logger import vre_logger as logger

class ComputeRepresentationMixin(ABC):
    """ComputeRepresentationMixin for representations that can be computed"""
    def __init__(self):
        self._output_size: tuple[int, int] | str | None = None
        self._batch_size: int | None = None
        self._output_dtype: str | np.dtype | None = None

    @abstractmethod
    def compute(self, video: VREVideo, ixs: list[int]):
        """given input data (batch of images/frewes), compute the output data of this representation"""

    @property
    def output_size(self) -> str | tuple[int, int]:
        """Returns the output size as a tuple/string or None if it's not explicitly set"""
        if self._output_size is None:
            logger.warning(f"[{self}] No output_size set, returning 'video_shape'. Call set_compute_params")
            return "video_shape"
        return self._output_size

    @output_size.setter
    def output_size(self, os: str | tuple[int, int]):
        assert isinstance(os, (str, tuple, list)), os
        if isinstance(os, (tuple, list)):
            assert len(os) == 2 and all(isinstance(x, int) and x > 0 for x in os), os
        if isinstance(os, str):
            assert os in ("native", "video_shape"), os
        self._output_size = os if isinstance(os, str) else tuple(os)

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

    @property
    def output_dtype(self) -> np.dtype:
        """the output dtype of the representation"""
        if self._output_dtype is None:
            return np.float32
        return self._output_dtype

    @output_dtype.setter
    def output_dtype(self, dtype: str | np.dtype):
        assert isinstance(dtype, (str, np.dtype)), dtype
        self._output_dtype = np.dtype(dtype)

    def set_compute_params(self, **kwargs):
        """set the compute parameters for the representation"""
        attributes = ["output_size", "batch_size", "output_dtype"]
        assert set(kwargs).issubset(attributes), (list(kwargs), attributes)
        for attr in attributes:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
