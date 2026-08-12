"""VREVideo module"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
from io import FileIO
import numpy as np

from .readers import FrameReader, build_frame_reader
from .writers import FrameWriter, build_frame_writer

class VREVideo(Iterable):
    """VREVideo -- A generic wrapper on top of a Video container"""
    def __init__(self, source: str | Path | FrameReader | np.ndarray | list[np.ndarray] | FileIO, **reader_kwargs):
        self.reader: FrameReader = build_frame_reader(source, **(reader_kwargs or {}))

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """the shape of a frame"""
        return self.shape[1:]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """The shape of this video"""
        return self.reader.shape

    @property
    def fps(self) -> float:
        """The FPS of this video"""
        return self.reader.fps

    @property
    def path(self) -> str:
        """The path of the video"""
        return self.reader.path

    def write(self, out_path: Path, start_frame: int = 0, end_frame: int | None = None, **writer_kwargs) -> FrameWriter:
        """writes this video to the disk"""
        if Path(out_path).suffix != "": # .mp4 etc.
            writer = build_frame_writer("ffmpeg")
        else:
            writer = build_frame_writer(**writer_kwargs)
        writer.write(self, out_path, start_frame, end_frame)
        return writer

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, ix: int | list[int] | np.ndarray | slice) -> np.ndarray:
        assert isinstance(ix, (int, list, np.ndarray, slice)), type(ix)
        return self.reader[ix]

    def __len__(self):
        return len(self.reader)

    def __repr__(self):
        return f"[VREVideo] {self.reader}"
