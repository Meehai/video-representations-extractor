"""Helper module to make sense of the arguments sent to vre.run()"""
from typing import Any
from pathlib import Path
import shutil

from .utils import is_dir_empty, parsed_str_type, VREVideo
from .logger import vre_logger as logger

RepresentationsSetup = dict[str, dict[str, Any]]

class VRERuntimeArgs:
    """
    VRE runtime args. Helper class to process the arguments sent to vre.run()
    Parameters:
    - video The video that this run operates on
    - representations The dictionary of representations that this run operates on
    - output_dir The directory used to output representations in this run.
    - start_frame The first frame to process (inclusive). If not provided, defaults to 0.
    - end_frame The last frame to process (inclusive). If not provided, defaults to len(video).
    - batch_size The batch size to use when processing the video. If not provided, defaults to 1.
    - export_npy Whether to export the npy files
    - export_png Whether to export the png files
    - output_dir_exists_mode What to do if the output dir already exists. Can be one of:
        - 'overwrite' Overwrite the output dir if it already exists
        - 'skip_computed' Skip the computed frames and continue from the last computed frame
        - 'raise' (default) Raise an error if the output dir already exists
    - exception_mode What to do when encountering an exception. It always writes the exception to 'exception.txt'.
        - 'skip_representation' Will stop the run of the current representation and start the next one
        - 'stop_execution' (default) Will stop the execution of VRE
    - output_size The resulted output shape in the npy/png directories. Valid options: a tuple (h, w), or a string:
        - 'native' whatever each representation outputs out of the box)
        - 'video_shape' (default) resizing to the video shape
    - n_threads_data_storer The number of workers used for the ThreadPool that stores data at each step. This is
    needed because storing data takes a lot of time sometimes, even more than the computation itself. Default: 1.
    - load_from_disk_if_computed If true, then it will try to read from the disk if a representation is computed.
    """
    def __init__(self, video: VREVideo, representations: dict[str, "Representation"], output_dir: Path,
                 start_frame: int | None, end_frame: int | None, batch_size: int, export_npy: bool, export_png: bool,
                 output_dir_exists_mode: str, exception_mode: str, output_size: str | tuple,
                 n_threads_data_storer: int, load_from_disk_if_computed: bool):
        assert batch_size >= 1, f"batch size must be >= 1, got {batch_size}"
        assert export_npy + export_png > 0, "At least one of export modes must be True"
        assert output_dir_exists_mode in ("overwrite", "skip_computed", "raise"), output_dir_exists_mode
        assert exception_mode in ("stop_execution", "skip_representation"), exception_mode
        if isinstance(output_size, str):
            assert output_size in ("native", "video_shape"), output_size
        else:
            assert len(output_size) == 2 and all(isinstance(x, int) for x in output_size), output_size
        if start_frame is None:
            start_frame = 0
            logger.warning("start frame not set, default to 0")

        assert isinstance(start_frame, int) and start_frame <= end_frame, (start_frame, end_frame)
        self.video = video
        self.output_dir = output_dir
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.batch_size = batch_size
        self.export_npy = export_npy
        self.export_png = export_png
        self.output_dir_exists_mode = output_dir_exists_mode
        self.exception_mode = exception_mode
        self.output_size = tuple(output_size) if not isinstance(output_size, str) else output_size
        self.n_threads_data_storer = n_threads_data_storer
        self.representations = representations
        self.load_from_disk_if_computed = load_from_disk_if_computed

        self.batch_sizes = {k: batch_size if r.batch_size is None else r.batch_size
                            for k, r in representations.items()}
        self.output_sizes = {k: output_size if r.output_size is None else r.output_size
                             for k, r in representations.items()}

        self.npy_paths: dict[str, list[Path]] = self._make_npy_paths() # {repr: [out_dir/npy/0.npz, ...]}
        self.png_paths: dict[str, list[Path]] = self._make_png_paths() # {repr: [out_dir/png/0.png, ...]}
        assert len(self.npy_paths) > 0 and len(self.png_paths) > 0, (len(self.npy_paths), len(self.png_paths))
        self._make_and_check_dirs()
        logger.info(self)

    def _make_npy_paths(self) -> dict[str, list[Path]]:
        npy_paths = {}
        for name in self.representations.keys():
            npy_base_dir = self.output_dir / name / "npy/"
            npy_paths[name] = [npy_base_dir / f"{t}.npz" for t in range(len(self.video))]
        return npy_paths

    def _make_png_paths(self) -> dict[str, list[Path]]:
        png_paths = {}
        for name in self.representations.keys():
            png_base_dir = self.output_dir / name / "png/"
            png_paths[name] = [png_base_dir / f"{t}.png" for t in range(len(self.video))]
        return png_paths

    def _make_and_check_dirs(self):
        for representation in self.representations:
            npy_dir: Path = self.npy_paths[representation][0].parent
            png_dir: Path = self.png_paths[representation][0].parent

            if self.export_npy and npy_dir.exists() and not is_dir_empty(npy_dir, "*.npz"):
                if self.output_dir_exists_mode == "overwrite":
                    logger.debug(f"Output dir '{npy_dir}' already exists, will overwrite it")
                    shutil.rmtree(npy_dir)
                else:
                    assert self.output_dir_exists_mode != "raise", \
                        f"'{npy_dir}' exists. Set --output_dir_exists_mode to 'overwrite' or 'skip_computed'"
            npy_dir.mkdir(exist_ok=True, parents=True)

            if self.export_png and png_dir.exists() and not is_dir_empty(png_dir, "*.png"):
                if self.output_dir_exists_mode == "overwrite":
                    logger.debug(f"Output dir '{png_dir}' already exists, will overwrite it")
                    shutil.rmtree(png_dir)
                else:
                    assert self.output_dir_exists_mode != "raise", \
                        f"'{png_dir}' exists. Set --output_dir_exists_mode to 'overwrite' or 'skip_computed'"
            png_dir.mkdir(exist_ok=True, parents=True)

    def __repr__(self):
        return f"""[{parsed_str_type(self)}]
- Video path: '{getattr(self.video, "file", "")}'
- Output dir: '{self.output_dir}')
- Representations ({len(self.representations)}): {", ".join(x for x in self.representations.keys())}
- Video shape: {self.video.shape} (FPS: {self.video.frame_rate:.2f})
- Output frames ({self.end_frame - self.start_frame}): [{self.start_frame} : {self.end_frame - 1}]
- Output shape: {self.output_size if self.output_size != "video_shape" else self.video.frame_shape[0:2]}
- Batch size: {self.batch_size}
- Export npy: {self.export_npy}
- Export png: {self.export_png}
- Exception mode: '{self.exception_mode}'
- Output dir exists mode: '{self.output_dir_exists_mode}'
- Thread pool workers for storing data (0 = using main thread): {self.n_threads_data_storer}
- Load from disk if computed: {self.load_from_disk_if_computed}
"""
