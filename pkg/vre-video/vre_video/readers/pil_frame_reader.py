"""pil_frame_reader - module for PIL-based functionality"""
from pathlib import Path
import os
import numpy as np
from vre_video.utils import natsorted, image_read
from .frame_reader import FrameReader

class PILFrameReader(FrameReader):
    """implements FrameReader using Pillow to read frame by frame"""
    def __init__(self, path: str | Path, fps: float | None = None, frames: list[int] | None = None):
        self._path = Path(path)
        self.frame_paths = natsorted(list(self._path.iterdir()), key=lambda p: p.name)
        self.data: list[np.ndarray] = [image_read(x) for x in self.frame_paths]
        _shp = self.data[0].shape
        assert all(x.shape == _shp for x in self.data), f"Not all shapes of all images are equal to first image: {_shp}"
        assert all(x.dtype == np.uint8 for x in self.data), f"Not all data is uint8: {set(x.dtype for x in self.data)}"
        self.data = np.array(self.data)
        self._fps = fps or float(os.getenv("VIDEO_FPS", "1"))

        # try to make the frames based on 1.png, 2.png, etc.
        if frames is None:
            try:
                frames = sorted([int(x.stem) for x in self.frame_paths])
            except Exception:
                pass

        self.frames = frames or list(range(len(self.data)))
        self.frames_ix = dict(zip(self.frames, range(len(self.frames)))) # {ix: frame}

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (len(self.data), *self.data[0].shape)

    @property
    def fps(self):
        return self._fps

    @property
    def path(self) -> str:
        return str(self._path)

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
