"""DataWriter module -- used to store binary (npz) or image (png) files given a representation output"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np

from .utils import image_write, is_dir_empty
from .representations import ReprOut, Representation, IORepresentationMixin, ComputeRepresentationMixin
from .logger import vre_logger as logger

Repr = Representation | IORepresentationMixin | ComputeRepresentationMixin

class DataWriter:
    """
    Class used to store one representation on disk. Single thread only, for multi-threaded use DataStorer.
    Parameters:
    - output_dir The directory used to output representation(s) in this run. Multiple Writers can exist at the same time
    - representation The (compute) representation that is outputted by this writer
    - output_dir_exists_mode What to do if the output dir already exists. Can be one of:
        - 'overwrite' Overwrite the output dir if it already exists
        - 'skip_computed' Skip the computed frames and continue from the last computed frame
        - 'raise' (default) Raise an error if the output dir already exists
    """
    def __init__(self, output_dir: Path, representation: Repr, output_dir_exists_mode: str):
        assert output_dir_exists_mode in ("overwrite", "skip_computed", "raise"), output_dir_exists_mode
        self.rep = representation
        assert self.rep.binary_format is not None or self.rep.image_format is not None, f"One must be set: {self.rep}"
        self.output_dir = output_dir
        self.output_dir_exists_mode = output_dir_exists_mode
        self._make_dirs()

    def write(self, y_repr: ReprOut, imgs: np.ndarray | None, batch: list[int]):
        """store the data in the right format"""
        assert y_repr is not None and y_repr.output is not None
        assert len(y_repr.output) == len(batch), f"{len(y_repr.output)} - {batch} ({len(batch)})"
        if self.rep.export_image:
            assert imgs is not None
            assert (shp := imgs.shape)[0] == len(batch) and shp[-1] == 3, \
                f"Expected {len(batch)} images ({batch=}), got {shp}"
            assert imgs.dtype in (np.uint8, np.uint16, np.uint32, np.uint64), imgs.dtype

        for i, t in enumerate(batch):
            if self.rep.export_binary:
                ext = self.rep.binary_format.value
                if (bin_path := self.output_dir / self.rep.name / ext / f"{t}.{ext}").exists():
                    logger.debug2(f"[{self.rep}] '{bin_path}' already exists. Skipping.")
                else:
                    disk_fmt = self.rep.to_disk_fmt(y_repr.output[i])
                    self.rep.save_to_disk(disk_fmt, bin_path)
                if (extra := y_repr.extra) is not None and len(y_repr.extra) > 0:
                    assert len(extra) == len(batch), f"Extra must be a list of len ({len(extra)}) = {len(batch)=}"
                    np.savez(bin_path.parent / f"{t}_extra.npz", extra[i])

            if self.rep.export_image:
                ext = self.rep.image_format.value
                if (img_path := self.output_dir / self.rep.name / ext / f"{t}.{ext}").exists():
                    logger.debug2(f"[{self.rep}] '{img_path}' already exists. Skipping.")
                else:
                    image_write(imgs[i], img_path)

    def all_batch_exists(self, frames: list[int]) -> bool:
        """true if all batch [l:r] exists on the disk"""
        def _path(writer: DataWriter, t: int, suffix: str) -> Path:
            return writer.output_dir / writer.rep.name / suffix / f"{t}.{suffix}"
        assert all(isinstance(frame, int) and frame >= 0 for frame in frames), (frames, self.rep)
        assert self.rep.export_binary or self.rep.export_image, self.rep
        for ix in frames:
            if self.rep.export_binary and not _path(self, ix, self.rep.binary_format.value).exists():
                return False
            if self.rep.export_image and not _path(self, ix, self.rep.image_format.value).exists():
                return False
        return True

    def to_dict(self):
        """A dict representation of this DataWriter and information about its ComputeRepresentation object"""
        return {
            "output_dir_exists_mode": self.output_dir_exists_mode,
            "batch_size": self.rep.batch_size, "export_binary": self.rep.export_binary,
            "binary_format": self.rep.binary_format.value, "compress": self.rep.compress,
            "export_image": self.rep.export_image, "image_format": self.rep.image_format.value,
            "dtype": str(self.rep.output_dtype), "size": self.rep.output_size,
        }

    def _make_dirs(self):
        def _make_and_check_one(output_dir: Path, format_type: str):
            fmt_dir = output_dir / self.rep.name / format_type # npy/npz/png etc.
            if fmt_dir.exists() and not is_dir_empty(fmt_dir, f"*.{format_type}"):
                if self.output_dir_exists_mode == "overwrite":
                    logger.debug(f"Output dir '{fmt_dir}' already exists, will overwrite it")
                    shutil.rmtree(fmt_dir)
                else:
                    assert self.output_dir_exists_mode != "raise", \
                        f"'{fmt_dir}' exists. Set --output_dir_exists_mode to 'overwrite' or 'skip_computed'"
            fmt_dir.mkdir(exist_ok=True, parents=True)
        if self.rep.export_binary:
            _make_and_check_one(self.output_dir, self.rep.binary_format.value)
        if self.rep.export_image:
            _make_and_check_one(self.output_dir, self.rep.image_format.value)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def __repr__(self):
        return f"""[DataWriter]
- Representation: '{self.rep}'
- Output dir: '{self.output_dir}' (exists mode: '{self.output_dir_exists_mode}')
- Export binary: {self.rep.export_binary} (binary format: {self.rep.binary_format.value}, compress: {self.rep.compress})
- Export image: {self.rep.export_image} (image format: {self.rep.image_format.value})"""
