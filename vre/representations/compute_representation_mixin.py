"""ComputeRepresentationMixin module"""
from enum import Enum
import numpy as np

from ..logger import vre_logger as logger
from ..base_mixin import BaseMixin
from .representation import ReprOut

class BinaryFormat(Enum):
    """types of binary outputs from a representation"""
    NOT_SET = "not-set"
    NPZ = "npz"
    NPY = "npy"

class ImageFormat(Enum):
    """types of image outputs from a representation"""
    NOT_SET = "not-set"
    PNG = "png"
    JPG = "jpg"

class ComputeRepresentationMixin(BaseMixin):
    """ComputeRepresentationMixin for representations that can be computed"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._output_size: tuple[int, int] | str | None = None
        self._batch_size: int | None = None
        self._output_dtype: str | np.dtype | None = None
        self._binary_format: BinaryFormat | None = None
        self._image_format: ImageFormat | None = None
        self._compress: bool | None = None

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
        self._output_size = os

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

    def cast(self, repr_data: ReprOut, dtype: str) -> ReprOut:
        """Cast the output of a self.make(frames) call into some other dtype"""
        if (np.issubdtype(self.output_dtype, np.integer) and np.issubdtype(dtype, np.floating) or
            np.issubdtype(self.output_dtype, np.floating) and np.issubdtype(dtype, np.integer)):
            raise TypeError(f"Cannot convert {self.output_dtype} to {dtype}")
        return ReprOut(output=repr_data.output.astype(dtype), extra=repr_data.extra)

    @property
    def binary_format(self) -> BinaryFormat:
        """the binary format of the representation"""
        if self._binary_format is None:
            logger.warning(f"[{self}] No binary_format set, returning not-set. Call set_compute_params")
            return BinaryFormat.NOT_SET
        return self._binary_format

    @binary_format.setter
    def binary_format(self, bf: BinaryFormat | str):
        assert isinstance(bf, (BinaryFormat, str)), f"Must be one of {[_bf.value for _bf in BinaryFormat]}. Got {bf}."
        bf = BinaryFormat(bf) if isinstance(bf, str) else bf
        self._binary_format = bf

    @property
    def export_binary(self) -> bool:
        """whether to export binary or not"""
        return self.binary_format != BinaryFormat.NOT_SET

    @property
    def image_format(self) -> ImageFormat:
        """the image format of the representation"""
        if self._image_format is None:
            logger.warning(f"[{self}] No image_format set, returning not-set. Call set_compute_params")
            return ImageFormat.NOT_SET
        return self._image_format

    @image_format.setter
    def image_format(self, imf: ImageFormat | str):
        assert isinstance(imf, (ImageFormat, str)), f"Must be one of {[_imf.value for _imf in ImageFormat]}. Got {imf}."
        imf = ImageFormat(imf) if isinstance(imf, str) else imf
        self._image_format = imf

    @property
    def export_image(self) -> bool:
        """whether to export image or not"""
        return self.image_format != ImageFormat.NOT_SET

    @property
    def compress(self) -> bool:
        """whether to compress the output or not"""
        if self._compress is None:
            logger.warning(f"[{self}] No compress set, returning True. Call set_compute_params")
            return True
        return self._compress

    @compress.setter
    def compress(self, c: bool):
        self._compress = c

    def set_compute_params(self, **kwargs):
        """set the compute parameters for the representation"""
        attributes = ["output_size", "batch_size", "output_dtype", "binary_format", "image_format", "compress"]
        for attr in attributes:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
