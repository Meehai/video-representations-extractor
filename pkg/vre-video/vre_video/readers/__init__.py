"""init file"""
from pathlib import Path
import sys
from io import IOBase
from pprint import pformat
import subprocess
import atexit
import numpy as np

from vre_video.utils import logger
from .frame_reader import FrameReader
from .pil_frame_reader import PILFrameReader
from .numpy_frame_reader import NumpyFrameReader
from .ffmpeg_frame_reader import FFmpegFrameReader, FFprobe
from .fd_frame_reader import FdFrameReader

def _fmt(source: str | Path | FrameReader | list[np.ndarray] | np.ndarray | IOBase) -> str:
    if isinstance(source, list):
        return f"source='list' (len: {len(source)})"
    if isinstance(source, np.ndarray):
        return f"source='np.ndarray' (shape: {source.shape})"
    if isinstance(source, str) and source == "-":
        return "stdin (-)"
    return str(source)

def _build_dev_video_ffmpeg(source: str) -> tuple[IOBase, tuple[int, int], float]:
    """creates a pipe that reads raw data from v4l2 (linux webcams) through ffmpeg"""
    ffprobe = FFprobe(source)
    if ffprobe.stream_info.get("codec_name", "") != "rawvideo":
        raise ValueError(f"Source: {source}. Stream info: {pformat(ffprobe.stream_info)}")
    height, width = ffprobe.shape[1:3]

    cmd = [
        "ffmpeg",
        "-fflags", "nobuffer+discardcorrupt",
        "-f", "v4l2",
        "-input_format", ffprobe.stream_info.get("pix_fmt", "rgb24"),
        "-framerate", str(ffprobe.fps),
        "-video_size", f"{width}x{height}",
        "-i", source,
        "-vf", "setpts=PTS-STARTPTS",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    logger.debug(f"Running '{' '.join(cmd)}'")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    atexit.register(process.terminate)
    return process.stdout, (height, width), ffprobe.fps

def build_frame_reader(source: str | Path | FrameReader | list[np.ndarray] | np.ndarray | IOBase,
                       **kwargs) -> FrameReader:
    """builds the frame reader given a source path for a VREVideo"""
    logger.debug(f"Building frame reader. Source: '{_fmt(source)}'. Args: {kwargs}")
    if isinstance(source, FrameReader):
        return source
    # pylint: disable=not-an-iterable
    if isinstance(source, np.ndarray) or (isinstance(source, list) and all(isinstance(x, np.ndarray) for x in source)):
        return NumpyFrameReader(source, **kwargs)
    if isinstance(source, IOBase):
        return FdFrameReader(source, **kwargs)
    if isinstance(source, str) and source == "-":
        return FdFrameReader(sys.stdin.buffer, **kwargs)
    if isinstance(source, str) and source.startswith("/dev"):
        data, resolution, fps = _build_dev_video_ffmpeg(source)
        return FdFrameReader(data, resolution, fps)
    if Path(source).is_dir():
        suffixes = list({x.suffix for x in Path(source).iterdir()})
        assert len(suffixes) == 1, suffixes
        if suffixes[0] in (".png", ".jpg"):
            return PILFrameReader(source, **kwargs)
        if suffixes[0] in (".npy", ".npz"):
            return NumpyFrameReader(source, **kwargs)
    # Otherwise, let ffmpeg handle it and it'll throw on errors.
    return FFmpegFrameReader(source, **kwargs)
