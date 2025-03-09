"""FakeVideo module"""
from pathlib import Path
import shutil
from overrides import overrides
from tqdm import trange
import numpy as np
from .vre_video import VREVideo
from ..utils import image_write

class FakeVideo(VREVideo):
    """
    FakeVideo -- class used to test representations with a given numpy array.
    If 'frames' is provided, then these are the frames indexes, i.e. video[ix] returns self.data[self.frames_ix[ix]]
    This can be used to create a FakeVideo out of a directory with files: [1.png, 10.png, 100.png]
    """
    def __init__(self, data: np.ndarray, fps: float, frames: list[int] | None = None):
        self.data = data
        super().__init__()
        assert len(data) > 0, "No data provided"
        self._fps = fps
        self.frames = list(range(len(data))) if frames is None else frames
        self.path = f"FakeVideo {self.data.shape}"
        assert 0 < len(self.frames) <= 1_000_000 # max 1M frames to keep it tight
        assert len(self.frames) == len(self.data), (self.frames, self.data)
        assert all(isinstance(frame, int) for frame in self.frames), self.frames
        assert sorted(set(self.frames)) == self.frames, self.frames
        self.frames_ix = dict(zip(self.frames, range(len(self.frames)))) # {ix: frame}

    @property
    @overrides
    def shape(self) -> tuple[int, int, int, int]:
        return self.data.shape

    @property
    @overrides
    def fps(self) -> float:
        return self._fps

    @overrides
    def write(self, out_path: Path, start_frame: int = 0, end_frame: int | None = None):
        if (pth := Path(out_path)).exists():
            shutil.rmtree(out_path, ignore_errors=True)
        pth.mkdir(parents=True)
        for ix in trange((len(self) if end_frame is None else end_frame) - start_frame, desc=pth.name):
            out_file = pth / f"{(start_frame + ix) * 1 / self.fps}.png"
            image_write(self[start_frame + ix], out_file)

    @overrides
    def get_one_frame(self, ix: int) -> np.ndarray:
        return self.data[self.frames_ix[ix]]

    def __len__(self) -> int:
        return self.frames[-1] + 1

    def __repr__(self):
        return f"[FakeVideo] FPS: {self.fps}. Len: {len(self.data)}. Frame shape: {self.data.shape[1:]}. FPS: "
