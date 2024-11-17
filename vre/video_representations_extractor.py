"""Video Representations Extractor module"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Any
import traceback
from tqdm import tqdm
import torch as tr

from .representations import Representation, ComputeRepresentationMixin, LearnedRepresentationMixin
from .representations.io_representation_mixin import IORepresentationMixin, load_from_disk_if_possible
from .vre_runtime_args import VRERuntimeArgs
from .data_writer import DataWriter
from .data_storer import DataStorer
from .metadata import Metadata
from .utils import VREVideo, now_fmt
from .logger import vre_logger as logger

def _make_batches(frames: list[int], batch_size: int) -> list[int]:
    """return 1D array [start_frame, start_frame+bs, start_frame+2*bs... end_frame]"""
    if batch_size > len(frames):
        logger.warning(f"batch size {batch_size} is larger than #frames to process {len(frames)}.")
        batch_size = len(frames)
    batches, n_batches = [], len(frames) // batch_size + (len(frames) % batch_size > 0)
    for i in range(n_batches):
        batches.append(frames[i * batch_size: (i + 1) * batch_size])
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

    def set_io_parameters(self, **kwargs) -> VideoRepresentationsExtractor:
        """Set the required params for all representations of IORepresentationMixin type"""
        for r in [_r for _r in self.representations.values() if isinstance(_r, IORepresentationMixin)]:
            r.set_io_params(**kwargs)
        return self

    def run(self, output_dir: Path, frames: list[int] | None = None, output_dir_exists_mode: str = "raise",
            exception_mode: str = "stop_execution", n_threads_data_storer: int = 0) -> Metadata:
        """
        The main loop of the VRE. This will run all the representations on the video and store results in the output_dir
        Parameters:
        - output_dir The directory where VRE will store the representations
        - frames The list of frames to run VRE for. If None, will export all the frames of the video
        - exception_mode What to do when encountering an exception. It always writes the exception to 'exception.txt'
          - 'skip_representation' Will stop the run of the current representation and start the next one
          - 'stop_execution' (default) Will stop the execution of VRE
        - n_threads_data_storer The number of threads used by the DataStorer
        Returns:
        - A dataframe with the run statistics for each representation
        """
        self._setup_logger(output_dir, now := now_fmt())
        runtime_args = VRERuntimeArgs(self.video, self.representations, frames, exception_mode, n_threads_data_storer)
        logger.info(runtime_args)
        self._metadata = Metadata(self.repr_names, runtime_args, self._logs_file.parent / f"run_metadata-{now}.json")
        for vre_repr in self._get_output_representations():
            assert isinstance(vre_repr, Representation), vre_repr
            assert isinstance(vre_repr, IORepresentationMixin), vre_repr
            assert vre_repr.binary_format is not None or vre_repr.image_format is not None, vre_repr
            data_writer = DataWriter(output_dir, representation=vre_repr, output_dir_exists_mode=output_dir_exists_mode)
            repr_had_exception = self._do_one_representation(data_writer, runtime_args) # vre_repr is part of writer
            if repr_had_exception and runtime_args.exception_mode == "stop_execution":
                raise RuntimeError(f"Representation '{vre_repr.name}' threw. Check '{self._logs_file}' for information")
            vre_repr.vre_free() if isinstance(vre_repr, LearnedRepresentationMixin) and vre_repr.setup_called else None
        self._end_run()
        return self._metadata

    # Private methods
    def _cleanup_one_representation(self, representation: Representation, data_storer: DataStorer):
        data_storer.join_with_timeout(timeout=30)
        if isinstance(representation, LearnedRepresentationMixin) and representation.setup_called:
            representation.vre_free()
        for dep in representation.dependencies:
            if isinstance(dep, LearnedRepresentationMixin) and dep.setup_called:
                dep.vre_free()

    def _compute_one_representation_batch(self, rep: Representation, batch: list[int], output_dir: Path):
        rep.data = None # Important to invalidate any previous results here
        load_from_disk_if_possible(rep, self.video, batch, output_dir)
        if rep.data is not None:
            return

        assert isinstance(rep, ComputeRepresentationMixin), rep
        rep.vre_setup() if isinstance(rep, LearnedRepresentationMixin) and not rep.setup_called else None
        for dep in rep.dependencies: # Note: hopefully toposorted...
            dep.data = None # TODO: make unit test with this (lingering .data from previous computation)
            load_from_disk_if_possible(dep, self.video, batch, output_dir)
            if dep.data is None:
                assert isinstance(dep, ComputeRepresentationMixin), dep
                dep.vre_setup() if isinstance(dep, LearnedRepresentationMixin) and not dep.setup_called else None
                dep.compute(self.video, ixs=batch)
        rep.compute(self.video, ixs=batch)
        assert rep.data is not None, f"{rep} {batch} {output_dir}"

    def _do_one_representation(self, data_writer: DataWriter, runtime_args: VRERuntimeArgs) -> bool:
        """main loop of each representation. Returns true if had exception, false otherwise"""
        data_storer = DataStorer(data_writer, runtime_args.n_threads_data_storer)
        logger.info(f"Running:\n{data_writer.rep}\n{data_storer}")
        rep: Representation | ComputeRepresentationMixin = data_writer.rep
        self._metadata.metadata["data_writers"][rep.name] = data_writer.to_dict()
        rep.output_size = self.video.frame_shape[0:2] if rep.output_size == "video_shape" else rep.output_size

        batches = _make_batches(runtime_args.frames, rep.batch_size)
        pbar = tqdm(total=runtime_args.n_frames, desc=f"[VRE] {rep.name} bs={rep.batch_size}")
        for i, batch in enumerate(batches):
            if i % runtime_args.store_metadata_every_n_iters == 0:
                self._metadata.store_on_disk()

            now = datetime.now()
            if data_writer.all_batch_exists(batch):
                self._metadata.add_time(rep.name, (datetime.now() - now).total_seconds(), len(batch))
                pbar.update(len(batch))
                continue

            try:
                tr.cuda.empty_cache() # might empty some unused memory, not 100% if needed.
                self._compute_one_representation_batch(rep=rep, batch=batch, output_dir=data_writer.output_dir)
                assert rep.data is not None, (rep, batch)
                rep.data.output_images = rep.make_images() if rep.export_image else None
                data_storer(rep.data)
            except Exception:
                self._log_error(f"\n[{rep.name} {rep.batch_size=} {batch=}] {traceback.format_exc()}\n")
                self._metadata.add_time(rep.name, 1 << 31, len(runtime_args.frames) - i * rep.batch_size)
                self._cleanup_one_representation(rep, data_storer)
                return True
            self._metadata.add_time(rep.name, (datetime.now() - now).total_seconds(), len(batch))
            pbar.update(len(batch))
        self._cleanup_one_representation(rep, data_storer)
        return False

    def _setup_logger(self, output_dir: Path, now_str: str):
        (logs_dir := output_dir / ".logs").mkdir(exist_ok=True, parents=True)
        self._logs_file = logs_dir / f"logs-{now_str}.txt"
        logger.add_file_handler(self._logs_file)

    def _log_error(self, msg: str):
        assert self._logs_file is not None, "_log_error can be called only after run() when self._log_file is set"
        self._logs_file.parent.mkdir(exist_ok=True, parents=True)
        open(self._logs_file, "a").write(f"{'=' * 80}\n{now_fmt()}\n{msg}\n{'=' * 80}")
        logger.debug(f"Error: {msg}")

    def _end_run(self):
        logger.remove_file_handler()
        self._metadata.store_on_disk()
        logger.info(f"Stored vre run log file at '{self._metadata.disk_location}")

    def _get_output_representations(self) -> list[IORepresentationMixin]:
        """given all the representations, keep those that actually export something. At least one is needed"""
        crs = [_r for _r in list(self.representations.values()) if isinstance(_r, IORepresentationMixin)]
        assert len(crs) > 0, f"No I/O Representation found in {self.repr_names}"
        out_r = [r for r in crs if r.export_binary or r.export_image]
        assert len(out_r) > 0, f"No output format set for any I/O Representation: {', '.join([r.name for r in crs])}"
        return out_r

    def __call__(self, *args, **kwargs) -> dict[str, Any]:
        return self.run(*args, **kwargs)

    def __str__(self) -> str:
        return f"""\n[VRE]
- Video: {self.video}
- Representations ({len(self.representations)}): [{", ".join([str(v) for v in self.representations.values()])}]"""

    def __repr__(self) -> str:
        return str(self)
