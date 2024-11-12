"""FakeVideo module"""
from typing import Union
import numpy as np
import ffmpeg

class FakeVideo:
    """FakeVideo -- class used to test representations with a given numpy array"""
    def __init__(self, data: np.ndarray, fps: float):
        assert len(data) > 0, "No data provided"
        self.data = data
        self.fps = fps
        self.frame_shape = data.shape[1:]
        self.file = f"FakeVideo {self.data.shape}"

    @property
    def shape(self):
        """returns the shape of the data"""
        return self.data.shape

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ix: int) -> np.ndarray:
        return self.data[ix]

    def __repr__(self):
        return f"[FakeVideo] FPS: {self.fps}. Len: {len(self.data)}. Frame shape: {self.data.shape[1:]}. FPS: "

class FFmpegVideo:
    """FFmpegVideo -- reads data from a video using ffmpeg"""
    def __init__(self, path, cache_len: int = 30):
        self.path = path
        self.probe = ffmpeg.probe(path)
        self.stream_info = next((stream for stream in self.probe["streams"] if stream["codec_type"] == "video"), None)
        self.fps = eval(self.stream_info["avg_frame_rate"]) # pylint: disable=eval-used
        self.width = int(self.stream_info["width"])
        self.height = int(self.stream_info["height"])
        self.total_frames = int(float(self.stream_info["nb_frames"])) if "nb_frames" in self.stream_info else None
        self.frame_shape = (self.height, self.width, 3)
        self.shape = (self.total_frames, *self.frame_shape)

        self.cache = []
        self.cache_max_len = cache_len
        self.cache_start_frame = None
        self.cache_end_frame = None
        self.video_process = None

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
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

    def _cache_frames(self, start_frame: int):
        """
        Cache all frames between the current keyframe and the next keyframe, starting at start_frame.
        """
        start_time = start_frame / self.fps
        self.cache = []
        self._start_ffmpeg_process(start_time)

        # Read frames until the end of the current keyframe range
        while True:
            in_bytes = self.video_process.stdout.read(self.width * self.height * 3)
            if not in_bytes or len(self.cache) > self.cache_max_len:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
            self.cache.append(frame)

        # Set cache boundaries in terms of frame numbers
        self.cache_start_frame = start_frame
        self.cache_end_frame = start_frame + len(self.cache)

    def get_frame_by_number(self, frame_number: int) -> np.ndarray:
        """Retrieve a frame from the video by frame number, using nearby frames caching."""
        if isinstance(frame_number, list):
            return np.array([self.get_frame_by_number(ix) for ix in frame_number], dtype=np.uint8)
        assert isinstance(frame_number, (int, list)), type(frame_number)
        assert 0 <= frame_number < self.total_frames, f"Frame out of bounds: {frame_number}. Len: {len(self)}"

        # Load new cache if the requested frame is outside the current cache range
        if self.cache_start_frame is None or not self.cache_start_frame <= frame_number < self.cache_end_frame:
            # keyframe_frame = int(frame_number - (frame_number % (1 / self.fps)))  # Nearest keyframe frame number
            self._cache_frames(frame_number)

        # Calculate the index within the cache
        cache_index = frame_number - self.cache_start_frame
        return self.cache[cache_index]

    def __repr__(self):
        return f"[FFmpegVideo] Path: {self.path}. FPS: {self.fps}. Len: {len(self)}. Frame shape: {self.frame_shape}."

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, frame_number: int) -> np.ndarray:
        return self.get_frame_by_number(frame_number)

    def __del__(self):
        """Clean up the ffmpeg process when done."""
        if self.video_process is not None:
            self.video_process.stdout.close()
            self.video_process.stderr.close()
            self.video_process.terminate()

VREVideo = Union[FFmpegVideo, FakeVideo]
