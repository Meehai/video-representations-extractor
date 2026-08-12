"""FFmpegFrameReader module"""
from pathlib import Path
from io import FileIO
from pprint import pformat
import os
import json
import subprocess
from overrides import overrides
import numpy as np

from .frame_reader import FrameReader
from ..utils import logger

class FFprobe:
    """FFprobe class. Calls the underlying ffprobe utility and extracts metadata about the video"""
    def __init__(self, path: str):
        self.path = path
        self.probe_data, self.stream_info = self._ffprobe_get_data()
        self.fps = self._build_fps()
        self.shape: tuple[int, int, int, int] = (self._build_total_frames(), *self._build_height_width(), 3)

    def _ffprobe_get_data(self) -> tuple[dict, dict]:
        cmd = [
            "ffprobe",
            "-v", "error",            # suppress unnecessary output
            "-select_streams", "v",   # select only video streams
            "-show_entries", "stream",  # show all stream fields
            "-show_entries", "format", # required for vp9
            "-of", "json",            # output format: JSON
            self.path,
        ]
        logger.debug(f"Running '{' '.join(cmd)}'")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)

        stream_info = next((stream for stream in probe_data.get("streams", []) # 1st video stream
                            if stream.get("codec_type") == "video"), None)
        return probe_data, stream_info

    def _build_fps(self) -> float:
        """builds the frames per second"""
        if "avg_frame_rate" in self.stream_info and self.stream_info["avg_frame_rate"] != "0/0":
            return eval(self.stream_info["avg_frame_rate"]) # pylint: disable=eval-used
        if "r_frame_rate" in self.stream_info:
            return eval(self.stream_info["r_frame_rate"]) # pylint: disable=eval-used
        raise ValueError(f"Cannot build FPS. Stream info from ffmpeg: {pformat(self.stream_info)}")

    def _build_total_frames(self) -> int:
        """returns the number of frames of the vifdeo"""
        if "nb_frames" in self.stream_info:
            return int(float(self.stream_info["nb_frames"]))
        if "duration" in self.stream_info:
            duration_s = float(self.stream_info["duration"])
            return int(duration_s * self.fps)
        if self.stream_info.get("codec_name", "") == "h264":
            if "tags" in self.stream_info:
                if "DURATION" in self.stream_info["tags"]:
                    duration_str = self.stream_info["tags"]["DURATION"]
                    h, m, s = [float(x) for x in duration_str.split(":")]
                    duration_s = h * 60 * 60 + m * 60 + s
                    return int(duration_s * self.fps)
                if "variant_bitrate" in self.stream_info["tags"]: # live stream feed like (Note: perhaps not only)
                    raise ValueError("For livestreams use '-' (stdin fd frame reader) and pipes")
        if self.stream_info.get("codec_name", "") == "vp9":
            if "format" in self.probe_data and "duration" in self.probe_data["format"]:
                duration_s = float(self.probe_data["format"]["duration"])
                return int(duration_s * self.fps)
        if self.stream_info.get("codec_name", "") == "rawvideo":
            return 2**31 # realtime video, infinite frames
        raise ValueError(f"Unknown video format. Stream info from ffmpeg: {pformat(self.stream_info)}")

    def _build_height_width(self) -> tuple[int, int]:
        """returns the (height, width) of this video"""
        width = int(self.stream_info["width"])
        height = int(self.stream_info["height"])

        if "side_data_list" in self.stream_info: # stream_info["side_data_list"][0]["rotation"] == -90 -> portrait
            for item in self.stream_info["side_data_list"]:
                if "rotation" in item and item["rotation"] == -90:
                    logger.debug("Portrait mode detected. Swapping width and height")
                    height, width = width, height
        return height, width

class FFmpegFrameReader(FrameReader):
    """FFmpegFrameReader -- reads data from a video using ffmpeg"""
    def __init__(self, path: str | Path, cache_len: int = int(os.getenv("FFMPEG_FRAME_READER_CACHE_LEN", "100"))):
        super().__init__()
        self._path = self._build_path(path)
        assert self._path.exists(), f"Video '{self._path}' doesn't exist. Use '-' (stdin frame reader) for live streams"
        self.ffprobe = FFprobe(self.path)

        self._fps = self.ffprobe.fps
        self._shape = self.ffprobe.shape

        self.cache = []
        self.cache_max_len = cache_len
        self.cache_start_frame = None
        self.cache_end_frame = None

        self.process: subprocess.Popen | None = None
        self.log_file: FileIO | None = None

    @property
    @overrides
    def path(self) -> str:
        return str(self._path)

    @property
    @overrides
    def shape(self) -> str:
        return self._shape

    @property
    @overrides
    def fps(self) -> str:
        return self._fps

    def get_one_frame(self, ix: int) -> np.ndarray:
        """Retrieve a frame from the video by frame number, using nearby frames caching."""
        assert isinstance(ix, int), type(ix)
        assert 0 <= ix < len(self), f"Frame out of bounds: {ix}. Len: {len(self)}"

        # Load new cache if the requested frame is outside the current cache range
        if self.cache_start_frame is None or not self.cache_start_frame <= ix < self.cache_end_frame:
            self._cache_frames(start_frame=ix)

        return self.cache[ix - self.cache_start_frame]

    def _build_path(self, path: str | Path) -> Path:
        """Builds the path. Can also be a youtube video, not just a local path, but yt_dlp must be installed"""
        if (s_path := str(path)).startswith("http") and (s_path.find("youtube") != -1 or s_path.find("youtu.be") != -1):
            from yt_dlp import YoutubeDL # pylint: disable=import-outside-toplevel, import-error
            tmpfile = f"/tmp/{path}.mp4"
            if not Path(tmpfile).exists():
                with YoutubeDL({"format": "bv*", "outtmpl": tmpfile, "quiet": True,
                                "no_warnings": True, "noprogress": True}) as ydl:
                    ydl.download([s_path])
            path = tmpfile
        return Path(path)

    def _start_ffmpeg_process(self, start_time: float) -> subprocess.Popen:
        """
        Start an ffmpeg process from the nearest keyframe to the requested start_time.
        This will load all frames from this keyframe to the next keyframe.
        """
        self._cleanup_if_needed()
        self.log_file = open(Path(self.path).with_suffix(".ffmpeg.log"), "w")
        cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", str(self.path),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:" # output to stdout
        ]
        logger.debug(f"Running '{' '.join(cmd)}' (log: {self.log_file.name})")
        self.process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self.log_file)

    def _cleanup_if_needed(self):
        if hasattr(self, "process") and self.process is not None: # hasattr in case the constructor throws on .exists()
            self.process.stdout.close()
            self.process.terminate()
        if hasattr(self, "log_file") and self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def _cache_frames(self, start_frame: int):
        """Start a fresh ffmpeg process at start_frame and cache up to cache_max_len+1 frames.
        (Reusing the running process between windows is a future optimization.)"""
        start_time = start_frame / self.fps
        self._start_ffmpeg_process(start_time)

        self.cache = []
        while len(self.cache) <= self.cache_max_len:
            in_bytes = self.process.stdout.read(self.shape[1] * self.shape[2] * 3)
            # If we reach the end of video (or the process somehow ended), also break.
            if not in_bytes:
                break

            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.shape[1], self.shape[2], 3])
            self.cache.append(frame)

        # Never leave a bogus cache_start == cache_end entry: if the stream ended before
        # ffprobe's count, fail loudly instead of IndexError'ing on the next access.
        if len(self.cache) == 0:
            raise EOFError(f"Stream ended at frame {start_frame} before the {len(self)} frames "
                           f"ffprobe reported (VFR/metadata mismatch?) -- cannot read frame {start_frame}")

        self.cache_start_frame = start_frame
        self.cache_end_frame = start_frame + len(self.cache)

    def __del__(self):
        self._cleanup_if_needed()

    def __getitem__(self, ix: int | list[int] | np.ndarray | slice) -> np.ndarray:
        if isinstance(ix, np.ndarray):
            return self[ix.tolist()]
        if isinstance(ix, list):
            return np.array([self[_ix] for _ix in ix])
        if isinstance(ix, slice):
            return np.array([self[_ix] for _ix in range(ix.start, ix.stop)])
        return self.get_one_frame(ix)
