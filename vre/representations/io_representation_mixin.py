"""IORepresentation: representations that are files on the disk and not computed from video or other representations"""
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import numpy as np

from vre.logger import vre_logger as logger
from vre.utils import image_resize_batch, MemoryData, DiskData
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

class IORepresentationMixin(ABC):
    """
    StoredRepresentation. These methods define the blueprint to store and load a representation from disk to RAM.
    Data transformations order:
    - DISK [.npz] -> load_from_disk() -> [disk_fmt] -> disk_to_memory_fmt() -> RAM [memory_fmt] (stored in .data)
    - RAM [memory_fmt] -> memory_to_disk_fmt() -> [disk_fmt] -> store_to_disk() -> DISK [.npz]
    The intermediate disk_fmt is only used for storage efficiency, while all the vre operations happen with memory_fmt
    For example: disk_fmt is a binary bool vector, while memory_fmt is a one-hot 2 channels map.
    """
    def __init__(self):
        self._binary_format: BinaryFormat | None = None
        self._image_format: ImageFormat | None = None
        self._compress: bool | None = None
        self._output_size: tuple[int, int] | str | None = None
        self._output_dtype: str | np.dtype | None = None

    @abstractmethod
    def load_from_disk(self, path: Path) -> DiskData:
        """Reads the data from the disk into disk_fmt"""

    @abstractmethod
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        """Transforms the data from disk_fmt into memory_fmt (usable in VRE)"""

    @abstractmethod
    def memory_to_disk_fmt(self, memory_data: MemoryData) -> DiskData:
        """Transformes the data from memory_fmt (usable in VRE) to disk_fmt"""

    @abstractmethod
    def save_to_disk(self, memory_data: MemoryData, path: Path):
        """Stores the disk_fmt data to disk"""

    @property
    def binary_format(self) -> BinaryFormat:
        """the binary format of the representation"""
        if self._binary_format is None:
            logger.warning(f"[{self}] No binary_format set, returning not-set. Call set_io_params")
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
            logger.warning(f"[{self}] No image_format set, returning not-set. Call set_io_params")
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
            logger.warning(f"[{self}] No compress set, returning True. Call set_io_params")
            return True
        return self._compress

    @compress.setter
    def compress(self, c: bool):
        self._compress = c

    @property
    def output_size(self) -> str | tuple[int, int]:
        """Returns the output size as a tuple/string or None if it's not explicitly set"""
        if self._output_size is None:
            logger.warning(f"[{self}] No output_size set, returning 'video_shape'. Call set_io_params")
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
    def output_dtype(self) -> np.dtype:
        """the output dtype of the representation"""
        if self._output_dtype is None:
            return np.float32
        return self._output_dtype

    @output_dtype.setter
    def output_dtype(self, dtype: str | np.dtype):
        assert isinstance(dtype, (str, np.dtype)), dtype
        self._output_dtype = np.dtype(dtype)

    def set_io_params(self, **kwargs):
        """set the IO parameters for the representation"""
        attributes = ["binary_format", "image_format", "compress", "output_size", "output_dtype"]
        res = ""
        assert set(kwargs).issubset(attributes), (list(kwargs), attributes)
        for attr in attributes:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
                res += f"\n-{attr}: {kwargs[attr]}"
        if len(res) > 0:
            logger.debug(f"[{self}] Set node specific 'IO' params: {res}")

    def resize(self, data: ReprOut, new_size: tuple[int, int]):
        """resizes the data. size is provided in (h, w)"""
        assert data is not None, "No data provided"
        interpolation = "nearest" if np.issubdtype(d := data.output.dtype, np.integer) or d == bool else "bilinear"
        output_images = None
        if data.output_images is not None:
            output_images = image_resize_batch(data.output_images, *new_size, interpolation="nearest")
        return ReprOut(frames=data.frames, key=data.key, extra=data.extra, output_images=output_images,
                       output=image_resize_batch(data.output, *new_size, interpolation=interpolation))
