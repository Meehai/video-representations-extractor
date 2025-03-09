"""FFmpegVideo module"""
from pathlib import Path
import ffmpeg
import numpy as np
from overrides import overrides
from tqdm import trange

from .vre_video import VREVideo

class FFmpegVideo(VREVideo):
    """FFmpegVideo -- reads data from a video using ffmpeg"""
    def __init__(self, path: Path, cache_len: int = 30):
        super().__init__()
        self.path = self._build_path(path)
        assert self.path.exists(), f"Video '{self.path}' doesn't exist"
        self.probe = ffmpeg.probe(self.path)
        self.stream_info = next((stream for stream in self.probe["streams"] if stream["codec_type"] == "video"), None)

        self._fps: float | None = None
        self._shape: tuple[int, int, int, int] | None = None

        self.cache = []
        self.cache_max_len = cache_len
        self.cache_start_frame = None
        self.cache_end_frame = None
        self.video_process = None
        self.write_process = None

    @property
    @overrides
    def shape(self) -> tuple[int, int, int, int]:
        if self._shape is None:
            width = int(self.stream_info["width"])
            height = int(self.stream_info["height"])
            total_frames = self._build_total_frames()
            self._shape = (total_frames, height, width, 3)
        return self._shape

    @property
    @overrides
    def fps(self) -> float:
        if self._fps is None:
            self._fps = eval(self.stream_info["avg_frame_rate"]) # pylint: disable=eval-used
        return self._fps

    @overrides
    def write(self, out_path: Path, start_frame: int = 0, end_frame: int | None = None):
        out_path = Path(out_path)
        assert self.write_process is None, self.write_process
        assert out_path.suffix == ".mp4", out_path
        assert isinstance(start_frame, int) and start_frame >= 0, start_frame

        self.write_process = (
            ffmpeg
            .input("pipe:0", format="rawvideo", pix_fmt="rgb24", s=f"{self.shape[2]}x{self.shape[1]}", r=self.fps)
            .output(str(out_path), pix_fmt="yuv420p", vcodec="libx264")
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=-3, pipe_stdout=-3) # -3 = subprocess.DEVNULL
        )

        try:
            for frame_ix in trange(start_frame, end_frame or len(self)):
                self.write_process.stdin.write(self[frame_ix].tobytes())
        finally:
            self.write_process.stdin.close()
            self.write_process.wait()
            self.write_process = None

    @overrides
    def get_one_frame(self, ix: int) -> np.ndarray:
        """Retrieve a frame from the video by frame number, using nearby frames caching."""
        assert isinstance(ix, int), type(ix)
        assert 0 <= ix < len(self), f"Frame out of bounds: {ix}. Len: {len(self)}"

        # Load new cache if the requested frame is outside the current cache range
        if self.cache_start_frame is None or not self.cache_start_frame <= ix < self.cache_end_frame:
            self._cache_frames(start_frame=ix)

        # Calculate the index within the cache
        return self.cache[ix - self.cache_start_frame]

    def _build_path(self, path: str | Path) -> Path:
        """Builds the path. Can also be a youtube video, not just a local path, but yt_dlp must be installed"""
        if (s_path := str(path)).startswith("http") and (s_path.find("youtube") != -1 or s_path.find("youtu.be") != -1):
            from yt_dlp import YoutubeDL # pylint: disable=import-outside-toplevel, import-error
            tmpfile = f"/tmp/{path}.mp4"
            if not Path(tmpfile).exists():
                with YoutubeDL({'format': 'bv*', 'outtmpl': tmpfile}) as ydl:
                    ydl.download([path])
            path = tmpfile
        return Path(path)

    def _build_total_frames(self) -> int:
        """returns the number of frames of the vifdeo"""
        if "nb_frames" in self.stream_info:
            return int(float(self.stream_info["nb_frames"]))
        if "duration" in self.stream_info:
            duration_s = float(self.stream_info["duration"])
            return int(duration_s * self.fps)
        if "codec_name" in self.stream_info and self.stream_info["codec_name"] == "h264":
            if "tags" in self.stream_info and "DURATION" in self.stream_info["tags"]:
                duration_str = self.stream_info["tags"]["DURATION"]
                h, m, s = [float(x) for x in duration_str.split(":")]
                duration_s = h * 60 * 60 + m * 60 + s
                return int(duration_s * self.fps)
        raise ValueError(f"Unknown video format. Stream info from ffmpeg: {self.stream_info}")

    def _start_ffmpeg_process(self, start_time):
        """
        Start an ffmpeg process from the nearest keyframe to the requested start_time.
        This will load all frames from this keyframe to the next keyframe.
        """
        # Close any existing process
        if self.video_process is not None:
            self.video_process.stdout.close()
            self.video_process.stderr.close()
            self.video_process.terminate()

        # Seek to the nearest keyframe before `start_time`
        self.video_process = (
            ffmpeg
            .input(self.path, ss=start_time)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run_async(pipe_stdout=True, pipe_stderr=True, pipe_stdin=True)
        )

    def _cache_frames(self, start_frame: int):
        """Cache all frames between the current keyframe and the next keyframe, starting at start_frame."""
        start_time = start_frame / self.fps
        self.cache = []
        self._start_ffmpeg_process(start_time)

        # Read frames until the end of the current keyframe range
        while True:
            in_bytes = self.video_process.stdout.read(self.shape[1] * self.shape[2] * 3)
            if not in_bytes or len(self.cache) > self.cache_max_len:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.shape[1], self.shape[2], 3])
            self.cache.append(frame)

        # Set cache boundaries in terms of frame numbers
        self.cache_start_frame = start_frame
        self.cache_end_frame = start_frame + len(self.cache)

    def __repr__(self):
        return f"[FFmpegVideo] Path: {self.path}. FPS: {self.fps}. Len: {len(self)}. Frame shape: {self.frame_shape}."

    def __len__(self) -> int:
        return self.shape[0]

    def __del__(self):
        """Clean up the ffmpeg process when done."""
        if hasattr(self, "video_process") and self.video_process is not None: # in case it throws in the constructor
            self.video_process.stdout.close()
            self.video_process.stderr.close()
            self.video_process.terminate()
