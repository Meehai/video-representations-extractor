"""Helper module to make sense of the arguments sent to vre.run()"""
from typing import Any

from .utils import parsed_str_type, VREVideo
from .logger import vre_logger as logger

RepresentationsSetup = dict[str, dict[str, Any]]

class VRERuntimeArgs:
    """
    VRE runtime args. Helper class to process the arguments sent to vre.run()
    Parameters:
    - video The video that this run operates on
    - representations The dictionary of representations that this run operates on
    - start_frame The first frame to process (inclusive). If not provided, defaults to 0.
    - end_frame The last frame to process (inclusive). If not provided, defaults to len(video).
    - batch_size The batch size to use when processing the video. If not provided, defaults to 1.
    - exception_mode What to do when encountering an exception. It always writes the exception to 'exception.txt'.
        - 'skip_representation' Will stop the run of the current representation and start the next one
        - 'stop_execution' (default) Will stop the execution of VRE
    - output_size The resulted output shape in the npz/png directories. Valid options: a tuple (h, w), or a string:
        - 'native' whatever each representation outputs out of the box)
        - 'video_shape' (default) resizing to the video shape
    - output_dtype: The dtype on which each representation is stored. If 'native', don't do anything
    - load_from_disk_if_computed If true, then it will try to read from the disk if a representation is computed.
    """
    def __init__(self, video: VREVideo, representations: dict[str, "Representation"],
                 start_frame: int | None, end_frame: int | None, batch_size: int,
                 exception_mode: str, output_size: str | tuple, load_from_disk_if_computed: bool):
        assert batch_size >= 1, f"batch size must be >= 1, got {batch_size}"
        assert exception_mode in ("stop_execution", "skip_representation"), exception_mode
        if isinstance(output_size, str):
            assert output_size in ("native", "video_shape"), output_size
        else:
            assert len(output_size) == 2 and all(isinstance(x, int) for x in output_size), output_size
        if start_frame is None:
            start_frame = 0
            logger.warning("start frame not set, default to 0")
        if end_frame is None:
            logger.warning(f"end frame not set, default to the last frame of the video: {len(video)}")
            end_frame = len(video)

        assert isinstance(start_frame, int) and start_frame <= end_frame, (start_frame, end_frame)
        self.video = video
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.batch_size = batch_size
        self.exception_mode = exception_mode
        self.output_size = tuple(output_size) if not isinstance(output_size, str) else output_size
        self.representations = representations
        self.load_from_disk_if_computed = load_from_disk_if_computed

        self.batch_sizes = {k: batch_size if r.batch_size is None else r.batch_size
                            for k, r in representations.items()}
        self.output_sizes = {k: output_size if r.output_size is None else r.output_size
                             for k, r in representations.items()}

    def __repr__(self):
        return f"""[{parsed_str_type(self)}]
- Video path: '{getattr(self.video, "file", "")}'
- Representations ({len(self.representations)}): {", ".join(x for x in self.representations.keys())}
- Video shape: {self.video.shape} (FPS: {self.video.frame_rate:.2f})
- Output frames ({self.end_frame - self.start_frame}): [{self.start_frame} : {self.end_frame - 1}]
- Output shape: {self.output_size if self.output_size != "video_shape" else self.video.frame_shape[0:2]}
- Batch size: {self.batch_size}
- Exception mode: '{self.exception_mode}'
- Load from disk if computed: {self.load_from_disk_if_computed}"""
