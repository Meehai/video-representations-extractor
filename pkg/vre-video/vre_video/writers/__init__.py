"""init file"""
import os
from .frame_writer import FrameWriter
from .pil_frame_writer import PILFrameWriter
from .numpy_frame_writer import NumpyFrameWriter
from .ffmpeg_frame_writer import FFmpegFrameWriter
from ..utils import logger

def build_frame_writer(writer: str | FrameWriter | None, **kwargs) -> FrameWriter:
    """builds the frame writer given a source path for a VREVideo"""
    if isinstance(writer, FrameWriter):
        return writer
    if writer is None:
        writer = os.getenv("VRE_VIDEO_DEFAULT_WRITER", "pillow")
        logger.debug(f"No writer provided. Defaulting to: '{writer}'. Change with 'VRE_VIDEO_DEFAULT_WRITER' env var.")
    if writer in ("pillow", "PIL"):
        return PILFrameWriter(**kwargs)
    if writer in ("numpy", "np"):
        return NumpyFrameWriter(**kwargs)
    if writer == "ffmpeg":
        return FFmpegFrameWriter(**kwargs)
    raise NotImplementedError(writer)
