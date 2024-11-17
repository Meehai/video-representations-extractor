"""IORepresentation: representations that are files on the disk and not computed from video or other representations"""
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import numpy as np

from vre.logger import vre_logger as logger
from vre.utils import VREVideo, image_resize_batch, MemoryData, DiskData
from .representation import Representation, ReprOut

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

    def set_io_params(self, **kwargs):
        """set the IO parameters for the representation"""
        attributes = ["binary_format", "image_format", "compress"]
        assert set(kwargs).issubset(attributes), (list(kwargs), attributes)
        for attr in attributes:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])

    def resize(self, data: ReprOut, new_size: tuple[int, int]):
        """resizes the data. size is provided in (h, w)"""
        assert data is not None, "No data provided"
        interpolation = "nearest" if np.issubdtype(d := data.output.dtype, np.integer) or d == bool else "bilinear"
        output_images = None
        if data.output_images is not None:
            output_images = image_resize_batch(data.output_images, *new_size, interpolation="nearest")
        return ReprOut(frames=data.frames, key=data.key, extra=data.extra, output_images=output_images,
                       output=image_resize_batch(data.output, *new_size, interpolation=interpolation))

def load_from_disk_if_possible(rep: Representation | IORepresentationMixin, video: VREVideo,
                               ixs: list[int], output_dir: Path):
    """loads (batched) data from disk if possible. Used in VRE main loop. TODO: integrate better in the class"""
    # TODO: see vre_with_stored_representations.py -- this needs to deepwalk if needed.
    assert isinstance(ixs, list) and all(isinstance(ix, int) for ix in ixs), (type(ixs), [type(ix) for ix in ixs])
    assert output_dir is not None and output_dir.exists(), output_dir
    npz_paths: list[Path] = [output_dir / rep.name / f"npz/{ix}.npz" for ix in ixs]
    extra_paths: list[Path] = [output_dir / rep.name / f"npz/{ix}_extra.npz" for ix in ixs]
    if any(not x.exists() for x in npz_paths): # partial batches are considered 'not existing' and overwritten
        return
    extras_exist = [x.exists() for x in extra_paths]
    assert (ee := sum(extras_exist)) in (0, (ep := len(extra_paths))), f"Found {ee}. Expected either 0 or {ep}"
    disk_data: DiskData = np.array([rep.load_from_disk(x) for x in npz_paths])
    extra = [np.load(x, allow_pickle=True)["arr_0"].item() for x in extra_paths] if ee == ep else None
    logger.debug2(f"[{rep}] Slice: [{ixs[0]}:{ixs[-1]}]. All data found on disk and loaded")
    rep.data = ReprOut(frames=video[ixs], output=rep.disk_to_memory_fmt(disk_data), extra=extra, key=ixs)
