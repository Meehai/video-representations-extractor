"""DataStorer module -- used to store npz/png files in a multi threaded way if needed"""
from threading import Thread
from multiprocessing import cpu_count
from queue import Queue
from time import time
from typing import Callable
import shutil
from pathlib import Path
import numpy as np

from .utils import image_write, is_dir_empty
from .representations import ReprOut
from .logger import vre_logger as logger

class DataWriter:
    """
    Class used to store the representations on disk. It supports multi-threading so we don't block the compute one
    Parameters:
    - output_dir The directory used to output representations in this run.
    - representations The list of representation names for this DataWriter
    - binary_format If set, it will save the representation as a binary file (npy/npz) + extra (npz always)
    - image_format If set, will store the representation as an image format (png/jpg). No extra.
    - output_dir_exists_mode What to do if the output dir already exists. Can be one of:
        - 'overwrite' Overwrite the output dir if it already exists
        - 'skip_computed' Skip the computed frames and continue from the last computed frame
        - 'raise' (default) Raise an error if the output dir already exists
    """
    def __init__(self, output_dir: Path, representations: list[str], output_dir_exists_mode: str,
                 binary_format: str | None, image_format: str | None, compress: bool = True):
        assert (binary_format is not None) + (image_format is not None) > 0, "At least one of format must be set"
        assert output_dir_exists_mode in ("overwrite", "skip_computed", "raise"), output_dir_exists_mode
        assert all(isinstance(r, str) for r in representations), representations
        assert binary_format is None or binary_format in ("npz", "npy", "npz_compressed"), binary_format
        assert image_format is None or image_format in ("png", "jpg"), image_format
        self.representations = representations
        self.output_dir = output_dir
        self.output_dir_exists_mode = output_dir_exists_mode
        self.export_binary = binary_format is not None
        self.export_image = image_format is not None
        self.binary_format = binary_format
        self.image_format = image_format
        self.compress = compress
        for r in representations:
            self._make_dirs_one_reprsentation(r)
        self.binary_func = self._make_binary_func()

    def write(self, name: str, y_repr: ReprOut, imgs: np.ndarray | None, l: int, r: int):
        """store the data in the right format"""
        assert name in self.representations, (name, self.representations)
        if self.export_image:
            assert (shp := imgs.shape)[0] == r - l and shp[-1] == 3, f"Expected {r - l} images ({l=} {r=}), got {shp}"
            assert imgs is not None
            assert imgs.dtype == np.uint8, imgs.dtype

        for i, t in enumerate(range(l, r)):
            if self.export_binary:
                if not (bin_path := self._path(name, t, self.binary_format)).exists(): # npy/npz_path
                    self.binary_func(y_repr.output[i], bin_path)
                    if (extra := y_repr.extra) is not None and len(y_repr.extra) > 0:
                        assert len(extra) == r - l, f"Extra must be a list of len ({len(extra)}) = batch_size ({r-l})"
                        np.savez(bin_path.parent / f"{t}_extra.npz", extra[i])
            if self.export_image:
                if not (img_path := self._path(name, t, self.image_format)).exists():
                    image_write(imgs[i], img_path)

    def all_batch_exists(self, representation: str, l: int, r: int) -> bool:
        """true if all batch [l:r] exists on the disk"""
        for ix in range(l, r):
            if self.export_binary and not self._path(representation, ix, self.binary_format).exists():
                return False
            if self.export_image and not self._path(representation, ix, self.image_format).exists():
                return False
        logger.debug2(f"Batch {representation}[{l}:{r}] exists on disk.")
        return True

    def _path(self, representation: str, t: int, suffix: str) -> Path:
        return self.output_dir / representation / suffix / f"{t}.{suffix}"

    def _make_dirs_one_reprsentation(self, representation: str):
        def _make_and_check_one(output_dir: Path, format_type: str):
            fmt_dir = output_dir / representation / format_type # npy/npz/png etc.
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
- Output dir: '{self.output_dir}' (exists mode: '{self.output_dir_exists_mode}')
- Export binary: {self.export_binary} (binary format: {self.binary_format}, compress: {self.compress})
- Export image: {self.export_image} (image format: {self.image_format})"""

class DataStorer:
    """
    Equivalent of DataLoader on top of a Dataset -> DataStorer on top of a DataWriter
    Parameters:
    - data_writer The DataWriter object used by this DataStorer (akin to Dataset and DataLoader)
    - n_threads_data_storer The number of workers used for the ThreadPool that stores data at each step. This is
    needed because storing data takes a lot of time sometimes, even more than the computation itself.
    """
    def __init__(self, data_writer: DataWriter, n_threads: int):
        assert n_threads >= 0, n_threads
        self.data_writer = data_writer
        self.n_threads = min(n_threads, cpu_count())
        self.threads: list[Thread] = []
        if n_threads >= 1:
            self.queue: Queue = Queue(maxsize=1) # maxisze=1 because we use many shared memory threads.
            for _ in range(n_threads):
                self.threads.append(thr := Thread(target=self._worker_fn, daemon=True, args=(self.queue, )))
                thr.start()
        logger.debug(f"[DataStorer] Set up with {n_threads} threads.")

    def _worker_fn(self, queue: Queue):
        while True: # Since these threads are daemon, they die when main thread dies. We could use None or smth to break
            self.data_writer(*queue.get())
            queue.task_done()

    def join_with_timeout(self, timeout: int):
        """calls queue.join() but throws after timeout seconds if it doesn't end"""
        if self.n_threads == 0:
            return
        logger.debug(f"Waiting for {self.queue.unfinished_tasks} leftover enqueued tasks")
        self.queue.all_tasks_done.acquire()
        try:
            endtime = time() + timeout
            while self.queue.unfinished_tasks:
                remaining = endtime - time()
                if remaining <= 0.0:
                    raise RuntimeError("Queue has not finished the join() before timeout")
                self.queue.all_tasks_done.wait(remaining)
        finally:
            self.queue.all_tasks_done.release()

    def __call__(self, name: str, y_repr: ReprOut, imgs: np.ndarray | None, l: int, r: int):
        assert isinstance(y_repr, ReprOut), f"{name=}, {type(y_repr)=}"
        if self.n_threads == 0:
            self.data_writer(name, y_repr, imgs, l, r)
        else:
            self.queue.put((name, y_repr, imgs, l, r), block=True, timeout=30)

    def __repr__(self):
        return f"""[DataStorer]
- Num threads: {self.n_threads} (0 = using main thread)
{self.data_writer}"""
