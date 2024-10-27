"""Video Representations Extractor module"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Any
import traceback
from tqdm import tqdm
import pandas as pd
import torch as tr

from .representations import Representation, LearnedRepresentationMixin, ComputeRepresentationMixin
from .vre_runtime_args import VRERuntimeArgs
from .data_writer import DataWriter
from .data_storer import DataStorer
from .metadata import Metadata
from .utils import VREVideo, now_fmt
from .logger import vre_logger as logger

def _make_batches(video: VREVideo, start_frame: int, end_frame: int, batch_size: int) -> list[int]:
    """return 1D array [start_frame, start_frame+bs, start_frame+2*bs... end_frame]"""
    if batch_size > end_frame - start_frame:
        logger.warning(f"batch size {batch_size} is larger than #frames to process [{start_frame}:{end_frame}].")
        batch_size = end_frame - start_frame
    last_one = min(end_frame, len(video))
    batches = list(range(start_frame, last_one, batch_size))
    batches = [*batches, last_one] if batches[-1] != last_one else batches
    return batches

class VideoRepresentationsExtractor:
    """Video Representations Extractor class"""

    def __init__(self, video: VREVideo, representations: dict[str, Representation]):
        """
        Parameters:
        - video The video we are performing VRE one
        - representations The dict of instantiated and topo sorted representations (or callable to instantiate them)
        """
        assert len(representations) > 0, "At least one representation must be provided"
        assert all(lambda x: isinstance(x, Representation) for x in representations.values()), representations
        self.video = video
        self.representations: dict[str, Representation] = representations
        self.repr_names = [r.name for r in representations.values()]
        self._logs_file: Path | None = None
        self._metadata: Metadata | None = None

    def set_compute_params(self, **kwargs: Any) -> VideoRepresentationsExtractor:
        """Set the required params for all representations of ComputeRepresentationMixin type"""
        for r in [_r for _r in self.representations.values() if isinstance(_r, ComputeRepresentationMixin)]:
            r.set_compute_params(**kwargs)
        return self

    def run(self, output_dir: Path, start_frame: int | None = None, end_frame: int | None = None,
            output_dir_exists_mode: str = "raise", exception_mode: str = "stop_execution",
            n_threads_data_storer: int = 0,
            load_from_disk_if_computed: bool = True) -> dict[str, Any]:
        """
        The main loop of the VRE. This will run all the representations on the video and store results in the output_dir
        See VRERuntimeArgs for parameters definition.
        Returns:
        - A dataframe with the run statistics for each representation
        """
        self._setup_logger(output_dir, now := now_fmt())
        runtime_args = VRERuntimeArgs(self.video, self.representations, start_frame, end_frame,
                                      exception_mode, load_from_disk_if_computed, n_threads_data_storer)
        logger.info(runtime_args)
        self._metadata = Metadata(self.repr_names, runtime_args, self._logs_file.parent / f"run_metadata-{now}.json")
        for vre_repr in self._get_output_representations():
            assert isinstance(vre_repr, Representation), vre_repr
            assert vre_repr.binary_format is not None or vre_repr.image_format is not None, vre_repr
            data_writer = DataWriter(output_dir, representation=vre_repr, output_dir_exists_mode=output_dir_exists_mode)
            repr_had_exception = self._do_one_representation(data_writer, runtime_args) # vre_repr is part of writer
            if repr_had_exception and runtime_args.exception_mode == "stop_execution":
                raise RuntimeError(f"Representation '{vre_repr.name}' threw. Check '{self._logs_file}' for information")
            vre_repr.vre_free() if isinstance(vre_repr, LearnedRepresentationMixin) and vre_repr.setup_called else None
        self._end_run()
        return self._metadata.metadata

    # Private methods
    def _do_one_representation(self, data_writer: DataWriter, runtime_args: VRERuntimeArgs) -> bool:
        """main loop of each representation. Returns true if had exception, false otherwise"""
        data_storer = DataStorer(data_writer, runtime_args.n_threads_data_storer)
        logger.info(f"Running:\n{data_writer.rep}\n{data_storer}")
        representation = data_writer.rep
        name, batch_size, output_size = representation.name, representation.batch_size, representation.output_size
        self._metadata.metadata["data_writers"][name] = data_writer.to_dict()

        batches = _make_batches(self.video, runtime_args.start_frame, runtime_args.end_frame, batch_size)
        left, right = batches[0:-1], batches[1:]
        pbar = tqdm(total=runtime_args.end_frame - runtime_args.start_frame, desc=f"[VRE] {name} bs={batch_size}")
        for i, (l, r) in enumerate(zip(left, right)): # main VRE loop
            if i % runtime_args.store_metadata_every_n_iters == 0:
                self._metadata.store_on_disk()

            now = datetime.now()
            try:
                tr.cuda.empty_cache() # might empty some unused memory, not 100% if needed.
                out_dir = data_writer.output_dir if runtime_args.load_from_disk_if_computed else None
                y_repr = representation.vre_make(video=self.video, ixs=slice(l, r), output_dir=out_dir)
                if output_size == "native":
                    y_repr_rsz = y_repr
                elif output_size == "video_shape":
                    y_repr_rsz = representation.resize(y_repr, self.video.frame_shape[0:2])
                else:
                    y_repr_rsz = representation.resize(y_repr, output_size)
                if y_repr_rsz.output.dtype != representation.output_dtype:
                    y_repr_rsz = representation.cast(y_repr_rsz, representation.output_dtype)
                imgs = representation.make_images(self.video[l: r], y_repr_rsz) if representation.export_image else None
                data_storer(y_repr_rsz, imgs, l, r)
            except Exception:
                self._log_error(f"\n[{name} {batch_size=} {l=} {r=}] {traceback.format_exc()}\n")
                self._metadata.add_time(name, 1 << 31, runtime_args.end_frame - l)
                data_storer.join_with_timeout(timeout=30)
                return True
            self._metadata.add_time(name, (datetime.now() - now).total_seconds(), r - l)
            pbar.update(r - l)
        data_storer.join_with_timeout(timeout=30)
        return False

    def _setup_logger(self, output_dir: Path, now_str: str):
        (logs_dir := output_dir / ".logs").mkdir(exist_ok=True, parents=True)
        self._logs_file = logs_dir / f"logs-{now_str}.txt"
        try:
            logger.add_file_handler(self._logs_file)
        except AssertionError: # TODO: FIX
            logger.remove_file_handler()
            logger.add_file_handler(self._logs_file)

    def _log_error(self, msg: str):
        assert self._logs_file is not None, "_log_error can be called only after run() when self._log_file is set"
        self._logs_file.parent.mkdir(exist_ok=True, parents=True)
        open(self._logs_file, "a").write(f"{'=' * 80}\n{now_fmt()}\n{msg}\n{'=' * 80}")
        logger.debug(f"Error: {msg}")

    def _end_run(self) -> pd.DataFrame:
        logger.remove_file_handler()
        self._metadata.store_on_disk()
        logger.info(f"Stored vre run log file at '{self._metadata.disk_location}")

    def _get_output_representations(self) -> list[ComputeRepresentationMixin]:
        """given all the compute representations, keep those that actually export something. At least one is needed"""
        crs = [_r for _r in list(self.representations.values()) if isinstance(_r, ComputeRepresentationMixin)]
        assert len(crs) > 0, f"No ComputeRepresentation found in {self.repr_names}"
        out_r = [r for r in crs if r.export_binary or r.export_image]
        assert len(out_r) > 0, f"No output format set for any ComputeRepresentation: {', '.join([r.name for r in crs])}"
        return out_r

    # pylint: disable=too-many-branches, too-many-nested-blocks
    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.run(*args, **kwargs)

    def __str__(self) -> str:
        return f"VRE ({len(self.representations)} representations). Video: '{self.video.file}' ({self.video.shape})"

    def __repr__(self) -> str:
        return str(self)
