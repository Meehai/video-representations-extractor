"""Helper module to make sense of the arguments sent to vre.run()"""
from typing import Any

from .utils import parsed_str_type, VREVideo
from .representations import Representation

RepresentationsSetup = dict[str, dict[str, Any]]

class VRERuntimeArgs:
    """
    VRE runtime args. Helper class to process the arguments sent to vre.run()
    Parameters:
    - video The video that this run operates on
    - representations The dictionary of representations that this run operates on
    - frames The list of frames to run VRE for. If None, will export all the frames of the video
    - exception_mode What to do when encountering an exception. It always writes the exception to 'exception.txt'
        - 'skip_representation' Will stop the run of the current representation and start the next one
        - 'stop_execution' (default) Will stop the execution of VRE
    - n_threads_data_storer The number of threads used by the DataStorer
    """
    def __init__(self, video: VREVideo, representations: dict[str, Representation],
                 frames: list[str] | None, exception_mode: str,
                 n_threads_data_storer: int, store_metadata_every_n_iters: int = 10):
        assert all(isinstance(r, Representation) for r in representations.values()), representations
        assert exception_mode in ("stop_execution", "skip_representation"), exception_mode
        frames = sorted(list(range(len(video))) if frames is None else frames)
        assert all(isinstance(x, int) for x in frames), frames
        assert 0 <= frames[0] <= frames[-1] < len(video), f"{frames[0]=}, {frames[-1]=}, {len(video)=}"

        self.video = video
        self.frames = frames
        self.exception_mode = exception_mode
        self.representations = representations
        self.n_threads_data_storer = n_threads_data_storer
        self.store_metadata_every_n_iters = store_metadata_every_n_iters

    @property
    def n_frames(self) -> int:
        """returns the number of frames to be computed by vre"""
        return len(self.frames)

    def to_dict(self) -> dict:
        """A dict representation of this runtime args. Used in Metadata() to be stored on disk during the run."""
        return {
            "video_path": str(getattr(self.video, "file", "")), "video_shape": f"{self.video.shape}",
            "video_fps": f"{self.video.fps:.2f}",
            "representations": [r.name for r in self.representations.values()],
            "frames": self.frames,
            "exception_mode": self.exception_mode,
            "n_threads_data_storer": self.n_threads_data_storer,
        }

    def __repr__(self):
        return f"""[{parsed_str_type(self)}]
- Video path: '{getattr(self.video, "file", "")}'
- Representations ({len(self.representations)}): {", ".join(x for x in self.representations.keys())}
- Video shape: {self.video.shape} (FPS: {self.video.fps:.2f})
- Output frames ({self.n_frames}): [{self.frames[0]} : {self.frames[-1]}]
- Exception mode: '{self.exception_mode}'
- DataStorer Threads: {self.n_threads_data_storer} (0 = only using main thread)
"""
