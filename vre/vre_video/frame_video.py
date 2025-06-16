"""FrameVideo module"""
from pathlib import Path
import shutil
from overrides import overrides
from tqdm import trange
from natsort import natsorted
import numpy as np
from .vre_video import VREVideo
from ..utils import image_write, image_read
from ..logger import vre_logger as logger

class FrameVideo(VREVideo):
    """
    FrameVideo -- class used to test representations with a given numpy array.
    If 'frames' is provided, then these are the frames indexes, i.e. video[ix] returns self.data[self.frames_ix[ix]]
    This can be used to create a FrameVideo out of a directory with files: [1.png, 10.png, 100.png]
    """
    def __init__(self, data: str | Path | np.ndarray, fps: float, frames: list[int] | None = None):
        super().__init__()
        self.raw_data = data
        self.data, self.frames = FrameVideo._build_data(data, frames)
        assert len(self.data) > 0, "No data provided"
        self._fps = fps
        assert 0 < len(self.frames) <= 1_000_000 # max 1M frames to keep it tight
        assert len(self.frames) == len(self.data), (self.frames, self.data)
        assert all(isinstance(frame, int) for frame in self.frames), self.frames
        assert sorted(set(self.frames)) == self.frames, self.frames
        self.frames_ix = dict(zip(self.frames, range(len(self.frames)))) # {ix: frame}

    @property
    def path(self) -> str:
        """The path of the video (if possible to compute)"""
        if isinstance(self.raw_data, (str, Path)):
            return f"FrameVideo {self.raw_data}"
        return f"FrameVideo {self.data.shape}"

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

    @staticmethod
    def _build_data(video_path: str | Path | np.ndarray, frames: list[str] | None) -> tuple[np.ndarray, list[str]]:
        if isinstance(video_path, np.ndarray):
            return video_path, frames or list(range(len(video_path)))
        all_files = natsorted([f for f in Path(video_path).iterdir() if f.is_file()], key=lambda p: p.name)
        suffixes = set(p.suffix for p in all_files)
        assert len(suffixes) == 1, f"Expected a single type of files in '{video_path}' found {suffixes}"
        assert next(iter(suffixes)) in (".png", ".npz"), suffixes
        frames = frames or [int(x.stem) for x in all_files]
        fn = {".png": image_read, ".npz": lambda p: np.load(p)["arr_0"], ".npy": np.load}[next(iter(suffixes))]
        raw_data = [fn(f) for f in all_files]
        assert all(x.shape == raw_data[0].shape for x in raw_data), f"Images shape differ in '{video_path}'"
        logger.debug(f"'video_path' is a directory. Assuming a directory of images. Found {len(raw_data)} images.")
        return np.array(raw_data, dtype=np.uint8), frames

    def __len__(self) -> int:
        return self.frames[-1] + 1

    def __repr__(self):
        return f"[FrameVideo] FPS: {self.fps}. Len: {len(self.data)}. Frame shape: {self.data.shape[1:]}. FPS: "
