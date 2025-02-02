"""FakeVideo module"""
from overrides import overrides
import numpy as np
from .vre_video import VREVideo

class FakeVideo(VREVideo):
    """
    FakeVideo -- class used to test representations with a given numpy array.
    If 'frames' is provided, then these are the frames indexes, i.e. video[ix] returns self.data[self.frames_ix[ix]]
    This can be used to create a FakeVideo out of a directory with files: [1.png, 10.png, 100.png]
    """
    def __init__(self, data: np.ndarray, fps: float, frames: list[int] | None = None):
        super().__init__()
        assert len(data) > 0, "No data provided"
        self.data = data
        self._fps = fps
        self.frames = list(range(len(data))) if frames is None else frames
        self.frame_shape = data.shape[1:]
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

    def __len__(self) -> int:
        return self.frames[-1] + 1

    def __getitem__(self, ix: int | list[int] | slice) -> np.ndarray:
        if isinstance(ix, list):
            return np.array([self[_ix] for _ix in ix])
        if isinstance(ix, slice):
            return np.array([self[_ix] for _ix in range(ix.start, ix.stop)])
        return self.data[self.frames_ix[ix]]

    def __repr__(self):
        return f"[FakeVideo] FPS: {self.fps}. Len: {len(self.data)}. Frame shape: {self.data.shape[1:]}. FPS: "
