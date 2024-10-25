"""DataWriter module -- used to store binary (npz) or image (png) files given a representation output"""
from typing import Callable
import shutil
from pathlib import Path
import numpy as np

from .utils import image_write, is_dir_empty
from .representations import ReprOut
from .logger import vre_logger as logger

class DataWriter:
    """
    Class used to store one representation on disk. Single thread only, for multi-threaded use DataStorer.
    Parameters:
    - output_dir The directory used to output representation(s) in this run. Multiple Writers can exist at the same time
    - representation The representation that is outputted by this writer
    - binary_format If set, it will save the representation as a binary file (npy/npz) + extra (npz always)
    - image_format If set, will store the representation as an image format (png/jpg). No extra.
    - output_dir_exists_mode What to do if the output dir already exists. Can be one of:
        - 'overwrite' Overwrite the output dir if it already exists
        - 'skip_computed' Skip the computed frames and continue from the last computed frame
        - 'raise' (default) Raise an error if the output dir already exists
    """
    def __init__(self, output_dir: Path, representation: str, output_dir_exists_mode: str,
                 binary_format: str | None, image_format: str | None, compress: bool = True):
        assert (binary_format is not None) + (image_format is not None) > 0, "At least one of format must be set"
        assert output_dir_exists_mode in ("overwrite", "skip_computed", "raise"), output_dir_exists_mode
        assert binary_format is None or binary_format in ("npz", "npy", "npz_compressed"), binary_format
        assert image_format is None or image_format in ("png", "jpg"), image_format
        self.representation = representation
        self.output_dir = output_dir
        self.output_dir_exists_mode = output_dir_exists_mode
        self.export_binary = binary_format is not None
        self.export_image = image_format is not None
        self.binary_format = binary_format
        self.image_format = image_format
        self.compress = compress
        self._make_dirs()
        self.binary_func = self._make_binary_func()

    def write(self, y_repr: ReprOut, imgs: np.ndarray | None, l: int, r: int):
        """store the data in the right format"""
        if self.export_image:
            assert imgs is not None
            assert (shp := imgs.shape)[0] == r - l and shp[-1] == 3, f"Expected {r - l} images ({l=} {r=}), got {shp}"
            assert imgs.dtype in (np.uint8, np.uint16, np.uint32, np.uint64), imgs.dtype

        for i, t in enumerate(range(l, r)):
            if self.export_binary:
                if not (bin_path := self._path(t, self.binary_format)).exists(): # npy/npz_path
                    self.binary_func(y_repr.output[i], bin_path)
                    if (extra := y_repr.extra) is not None and len(y_repr.extra) > 0:
                        assert len(extra) == r - l, f"Extra must be a list of len ({len(extra)}) = batch_size ({r-l})"
                        np.savez(bin_path.parent / f"{t}_extra.npz", extra[i])
            if self.export_image:
                if not (img_path := self._path(t, self.image_format)).exists():
                    image_write(imgs[i], img_path)

    def all_batch_exists(self, l: int, r: int) -> bool:
        """true if all batch [l:r] exists on the disk"""
        assert isinstance(l, int) and isinstance(r, int) and 0 <= l < r, (l, r, type(l), type(r))
        for ix in range(l, r):
            if self.export_binary and not self._path(ix, self.binary_format).exists():
                return False
            if self.export_image and not self._path(ix, self.image_format).exists():
                return False
        logger.debug2(f"Batch {self.representation}[{l}:{r}] exists on disk.")
        return True

    def _path(self, t: int, suffix: str) -> Path:
        return self.output_dir / self.representation / suffix / f"{t}.{suffix}"

    def _make_dirs(self):
        def _make_and_check_one(output_dir: Path, format_type: str):
            fmt_dir = output_dir / self.representation / format_type # npy/npz/png etc.
            if fmt_dir.exists() and not is_dir_empty(fmt_dir, f"*.{format_type}"):
                if self.output_dir_exists_mode == "overwrite":
                    logger.debug(f"Output dir '{fmt_dir}' already exists, will overwrite it")
                    shutil.rmtree(fmt_dir)
                else:
                    assert self.output_dir_exists_mode != "raise", \
                        f"'{fmt_dir}' exists. Set --output_dir_exists_mode to 'overwrite' or 'skip_computed'"
            fmt_dir.mkdir(exist_ok=True, parents=True)
        if self.export_binary:
            _make_and_check_one(self.output_dir, self.binary_format)
        if self.export_image:
            _make_and_check_one(self.output_dir, self.image_format)

    def _make_binary_func(self) -> Callable[[object, Path], None] | None:
        if self.export_binary is None:
            return None
        if self.binary_format == "npy":
            assert self.compress is False
            return lambda obj, path: np.save(path, obj)
        if self.binary_format == "npz":
            if self.compress:
                return lambda obj, path: np.savez_compressed(path, obj)
            return lambda obj, path: np.savez(path, obj)
        raise ValueError(f"Unknown binary format: '{self.binary_format}'")

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def __repr__(self):
        return f"""[DataWriter]
- Representation: '{self.representation}'
- Output dir: '{self.output_dir}' (exists mode: '{self.output_dir_exists_mode}')
- Export binary: {self.export_binary} (binary format: {self.binary_format}, compress: {self.compress})
- Export image: {self.export_image} (image format: {self.image_format})"""
