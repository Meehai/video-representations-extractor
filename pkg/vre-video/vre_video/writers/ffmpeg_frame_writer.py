"""ffmpeg_frame_writer - Module that implements frame writing using ffmpeg"""
from pathlib import Path
from io import FileIO
import os
import subprocess
from tqdm import trange
import numpy as np

from .frame_writer import FrameWriter
from ..utils import logger

class FFmpegFrameWriter(FrameWriter):
    """FFmpegFrameWriter implementation that writes to disk a video using ffmpeg process"""
    def write(self, video: "VREVideo", out_path: Path, start_frame: int = 0, end_frame: int | None = None):
        out_path = Path(out_path)
        assert out_path.suffix == ".mp4", out_path
        assert isinstance(start_frame, int) and start_frame >= 0, start_frame

        process, logfile = self._start_ffmpeg_write_process(video, out_path)

        assert not out_path.exists(), out_path
        out_path.parent.mkdir(exist_ok=True, parents=True)
        start_frame = start_frame or 0
        end_frame = end_frame or len(video)
        assert start_frame >= 0 and end_frame <= len(video), (start_frame, end_frame)

        try:
            for i in trange(start_frame, end_frame, disable=os.getenv("VRE_VIDEO_PBAR", "1") == "0",
                            desc=f"[FFmpegFrameWriter] {out_path}"):
                frame: np.ndarray = video[i]
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
            process.wait()
            logfile.close()

    def _start_ffmpeg_write_process(self, video: "VREVideo", out_path: Path) -> tuple[subprocess.Popen, FileIO]:
        log_file = open(out_path.with_suffix(".ffmpeg.log"), "w")
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output file if it exists
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{video.shape[2]}x{video.shape[1]}",
            "-r", str(video.fps),
            "-i", "pipe:0",             # input comes from stdin
            "-an",                      # no audio
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_path)
        ]
        logger.debug(f"Running '{' '.join(cmd)}' (log: {log_file.name})")
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_file)
        return process, log_file
