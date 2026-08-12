"""numpy_frame_reader - module for NumPy-based functionality"""
from pathlib import Path
import os
import numpy as np
from vre_video.utils import natsorted
from .frame_reader import FrameReader

class NumpyFrameReader(FrameReader):
    """Implements FrameReader using NumPy arrays, from memory or from disk (.npy/.npz files)"""
    def __init__(self, data: list[np.ndarray] | np.ndarray | str | Path, fps: float | None = None,
                 frames: list[int] | None = None):
        self.data, self.frames = NumpyFrameReader._build_data(data, frames)
        self._fps = fps or float(os.getenv("VIDEO_FPS", "1"))
        self.frames = self.frames or list(range(len(self.data)))
        self.frames_ix = dict(zip(self.frames, range(len(self.frames)))) # {ix: frame}
        self._path = str(data) if isinstance(data, (str, Path)) else "in memory"

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.data.shape

    @property
    def fps(self):
        return self._fps

    @property
    def path(self) -> str:
        return str(self._path)

    @staticmethod
    def _build_data(data: list[np.ndarray] | np.ndarray | str | Path,
                    frames: list[int] | None = None) -> tuple[np.ndarray, list[int]]:
        if isinstance(data, np.ndarray):
            assert data.ndim == 4, f"Expected 4D NumPy array (N, H, W, C), got shape {data.shape}"
            assert data.dtype == np.uint8, f"Expected dtype uint8, got {data.dtype}"
            return data, frames

        if isinstance(data, list):
            _shp = data[0].shape
            assert all(x.shape == _shp for x in data), f"Not all shapes match first image: {_shp}"
            assert all(x.dtype == np.uint8 for x in data), f"Not all data is uint8: {set(x.dtype for x in data)}"
            return np.array(data), frames

        if isinstance(data, (str, Path)):
            path = Path(data)
            assert path.exists() and path.is_dir(), f"Invalid path: '{path}'"

            files = natsorted([p for p in path.iterdir() if p.suffix in {".npy", ".npz"}], key=lambda p: p.name)
            suffixes = {f.suffix for f in files}
            assert len(suffixes) == 1, f"Directory must contain only one type of file (.npy or .npz), found: {suffixes}"

            if files[0].suffix == ".npy":
                data_lst: list[np.ndarray] = [np.load(f) for f in files]
            else:
                data_lst: list[np.ndarray] = [np.load(f)["arr_0"] for f in files]

            # try to make the frames based on 1.npz, 2.npz, etc.
            if frames is None:
                try:
                    frames = sorted([int(x.stem) for x in files])
                except Exception:
                    pass

            assert all(x.shape == data_lst[0].shape for x in data_lst), f"Shapes differ: expected {data_lst[0].shape}"
            assert all(x.dtype == np.uint8 for x in data_lst), f"Dtype mismatch: {[x.dtype for x in data_lst]}"
            return np.array(data_lst), frames

        raise TypeError(f"Data must be a list of np.ndarray, a 4D np.ndarray, or a directory path: Got {type(data)}.")

    def __getitem__(self, ix: int | list[int] | np.ndarray | slice) -> np.ndarray:
        if isinstance(ix, int):
            return self.data[self.frames_ix[ix]]
        if isinstance(ix, np.ndarray):
            ix = ix.tolist()
        if isinstance(ix, slice):
            ix = list(range(ix.start, ix.stop))
        ix_transformed = [self.frames_ix[_ix] for _ix in ix] # support for sparse frames (VRE)
        return self.data[ix_transformed]

    def __len__(self):
        return self.frames[-1] + 1
