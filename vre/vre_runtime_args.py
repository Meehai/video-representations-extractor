"""Helper module to make sense of the arguments sent to vre.run()"""
from typing import Any

from .utils import parsed_str_type, VREVideo
from .logger import vre_logger as logger
from .representations import Representation

RepresentationsSetup = dict[str, dict[str, Any]]

class VRERuntimeArgs:
    """
    VRE runtime args. Helper class to process the arguments sent to vre.run()
    Parameters:
    - video The video that this run operates on
    - representations The dictionary of representations that this run operates on
    - start_frame The first frame to process (inclusive). If not provided, defaults to 0.
    - end_frame The last frame to process (inclusive). If not provided, defaults to len(video).
    - exception_mode What to do when encountering an exception. It always writes the exception to 'exception.txt'.
        - 'skip_representation' Will stop the run of the current representation and start the next one
        - 'stop_execution' (default) Will stop the execution of VRE
    - load_from_disk_if_computed If true, then it will try to read from the disk if a representation is computed.
    - n_threads_data_storer The number of threads used by the DataStorer
    """
    def __init__(self, video: VREVideo, representations: dict[str, Representation],
                 start_frame: int | None, end_frame: int | None, exception_mode: str,
                 load_from_disk_if_computed: bool, n_threads_data_storer: int,
                 store_metadata_every_n_iters: int = 10):
        assert exception_mode in ("stop_execution", "skip_representation"), exception_mode
        if start_frame is None:
            start_frame = 0
            logger.warning("start frame not set, default to 0")
        if end_frame is None:
            logger.warning(f"end frame not set, default to the last frame of the video: {len(video)}")
            end_frame = len(video)

        assert isinstance(start_frame, int) and start_frame <= end_frame, (start_frame, end_frame)
        assert (end_frame - start_frame) <= len(video), f"{start_frame=}, {end_frame=}, {len(video)=}"
        self.video = video
        self.start_frame = start_frame
        self.end_frame = end_frame
        # self.batch_size = min(batch_size, end_frame - start_frame)
        self.exception_mode = exception_mode
        self.representations = representations
        self.load_from_disk_if_computed = load_from_disk_if_computed
        self.n_threads_data_storer = n_threads_data_storer
        self.store_metadata_every_n_iters = store_metadata_every_n_iters

    def to_dict(self) -> dict:
        """A dict representation of this runtime args. Used in Metadata() to be stored on disk during the run."""
        return {
            "video_path": getattr(self.video, "file", ""), "video_shape": f"{self.video.shape}",
            "video_fps": f"{self.video.frame_rate:.2f}",
            "representations": [r.name for r in self.representations.values()],
            "frames": (self.start_frame, self.end_frame),
            "exception_mode": self.exception_mode, "load_from_disk_if_computed": self.load_from_disk_if_computed,
            "n_threads_data_storer": self.n_threads_data_storer,
        }

    def __repr__(self):
# - Output sizes: {self.output_sizes}
# - Batch size: {self.batch_size}
        return f"""[{parsed_str_type(self)}]
- Video path: '{getattr(self.video, "file", "")}'
- Representations ({len(self.representations)}): {", ".join(x for x in self.representations.keys())}
- Video shape: {self.video.shape} (FPS: {self.video.frame_rate:.2f})
- Output frames ({self.end_frame - self.start_frame}): [{self.start_frame} : {self.end_frame - 1}]
- Exception mode: '{self.exception_mode}'
- Load from disk if computed: {self.load_from_disk_if_computed}
- #threads DataStorer: {self.n_threads_data_storer} (0 = only using main thread)
"""
