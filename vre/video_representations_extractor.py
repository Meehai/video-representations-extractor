"""Video Representations Extractor module"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import traceback
from tqdm import tqdm
import pandas as pd
import torch as tr

from .representations import Representation, LearnedRepresentationMixin
from .vre_runtime_args import VRERuntimeArgs
from .data_writer import DataStorer, DataWriter
from .utils import VREVideo, took, make_batches, now_fmt
from .logger import vre_logger as logger

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
        self._data_storer: DataStorer | None = None
        self._logs_file: Path | None = None
        self._metadata: dict | None = None

    def _do_one_representation(self, representation: Representation, runtime_args: VRERuntimeArgs) -> list[float]:
        """main loop of each representation."""
        name = representation.name
        batch_size, output_size = runtime_args.batch_sizes[name], runtime_args.output_sizes[name]
        output_dir, export_image  = self._data_storer.data_writer.output_dir, self._data_storer.data_writer.export_image

        # call vre_setup here so expensive representations get lazy deep instantiated (i.e. models loading)
        try:
            representation.vre_setup() if isinstance(representation, LearnedRepresentationMixin) else None # device
        except Exception:
            self._log_error(f"\n[{name} {batch_size=}] {traceback.format_exc()}\n")
            return [1 << 31] * (runtime_args.end_frame - runtime_args.start_frame)

        batches = make_batches(self.video, runtime_args.start_frame, runtime_args.end_frame, batch_size)
        left, right = batches[0:-1], batches[1:]
        repr_stats: list[float] = []
        pbar = tqdm(total=runtime_args.end_frame - runtime_args.start_frame, desc=f"[VRE] {name} bs={batch_size}")
        for l, r in zip(left, right): # main VRE loop
            if self._data_storer.data_writer.all_batch_exists(representation.name, l, r):
                pbar.update(r - l)
                repr_stats.extend(took(datetime.now(), l, r))
                continue

            now = datetime.now()
            try:
                tr.cuda.empty_cache() # might empty some unused memory, not 100% if needed.
                out_dir = output_dir if runtime_args.load_from_disk_if_computed else None
                y_repr = representation.vre_make(video=self.video, ixs=slice(l, r), output_dir=out_dir)
                if output_size == "native":
                    y_repr_rsz = y_repr
                elif output_size == "video_shape":
                    y_repr_rsz = representation.resize(y_repr, self.video.frame_shape[0:2])
                else:
                    y_repr_rsz = representation.resize(y_repr, output_size)
                imgs = representation.make_images(self.video[l: r], y_repr_rsz) if export_image else None
                if representation.output_dtype != "native": # TODO: test
                    y_repr_rsz.output = y_repr_rsz.output.astype(representation.output_dtype)
                self._data_storer(name, y_repr_rsz, imgs, l, r)
            except Exception:
                self._log_error(f"\n[{name} {batch_size=} {l=} {r=}] {traceback.format_exc()}\n")
                repr_stats.extend([1 << 31] * (runtime_args.end_frame - l))
                break
            # update the statistics and the progress bar
            repr_stats.extend(took(now, l, r))
            pbar.update(r - l)
        return repr_stats

    def run(self, output_dir: Path, start_frame: int | None = None, end_frame: int | None = None, batch_size: int = 1,
            binary_format: str | None = None, image_format: str | None = None, compress: bool = True,
            output_dir_exists_mode: str = "raise",
            exception_mode: str = "stop_execution", output_size: str | tuple = "video_shape",
            n_threads_data_storer: int = 0, load_from_disk_if_computed: bool = True) -> pd.DataFrame:
        """
        The main loop of the VRE. This will run all the representations on the video and store results in the output_dir
        See VRERuntimeArgs for parameters definition.
        Returns:
        - A dataframe with the run statistics for each representation
        """
        self._setup_logger(output_dir, now := now_fmt())
        runtime_args = VRERuntimeArgs(self.video, self.representations, start_frame, end_frame, batch_size,
                                      exception_mode, output_size, load_from_disk_if_computed)
        data_writer = DataWriter(output_dir, [r.name for r in self.representations.values()],
                                 output_dir_exists_mode=output_dir_exists_mode, binary_format=binary_format,
                                 image_format=image_format, compress=compress)
        self._data_storer = DataStorer(data_writer, n_threads_data_storer)
        logger.info(f"{runtime_args}\n{self._data_storer}")
        self._metadata = {"run_stats": {}}
        for name, vre_repr in self.representations.items():
            repr_res = self._do_one_representation(vre_repr, runtime_args)
            if repr_res[-1] == 1 << 31 and runtime_args.exception_mode == "stop_execution":
                raise RuntimeError(f"Representation '{name}' threw. Check '{self._logs_file}' for information")
            self._metadata["run_stats"][name] = repr_res
            vre_repr.vre_free() if isinstance(vre_repr, LearnedRepresentationMixin) else None # free device
        self._data_storer.join_with_timeout(timeout=30)
        self._store_metadata_and_end_run(runtime_args.start_frame, runtime_args.end_frame, now)
        return self._metadata

    # Private methods
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

    def _store_metadata_and_end_run(self, start_frame: int, end_frame: int, now_str: str) -> pd.DataFrame:
        logger.remove_file_handler()
        self._metadata["frames"] = (start_frame, end_frame)
        metadata_file = self._logs_file.parent / f"run_metadata-{now_str}.json"
        json.dump(self._metadata, open(metadata_file, "w"), indent=4)
        logger.info(f"Stored vre run log file at '{metadata_file}")

    # pylint: disable=too-many-branches, too-many-nested-blocks
    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.run(*args, **kwargs)

    def __str__(self) -> str:
        return f"VRE ({len(self.representations)} representations). Video: '{self.video.file}' ({self.video.shape})"

    def __repr__(self) -> str:
        return str(self)
