"""Video Repentations Extractor module"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os
import traceback
from tqdm import tqdm
import torch as tr
import numpy as np
from vre_video import VREVideo

from .representations import Representation, LearnedRepresentationMixin, RepresentationsList, IORepresentationMixin
from .vre_runtime_args import VRERuntimeArgs
from .data_writer import DataWriter
from .data_storer import DataStorer
from .run_metadata import RunMetadata
from .representation_metadata import RepresentationMetadata
from .utils import now_fmt, make_batches, ReprOut, DiskData, SummaryPrinter, random_chars, MemoryData
from .logger import vre_logger as logger

# TODO: split in 2 classes ?
# vre_batched = VREBatched(VREVideo, RepresentationsList); vre_batch.run(...)
# vre_streaming = VREStreaming(VREVIdeo(??) RepresentationsList); vre_streaming[0:5]

class VideoRepresentationsExtractor:
    """Video Representations Extractor class"""

    def __init__(self, video: VREVideo, representations: RepresentationsList | list[Representation]):
        """
        Parameters:
        - video The video we are performing VRE one
        - representations The list instantiated representations. Must be topo-sortable based on name and deps.
        """
        assert isinstance(video, VREVideo), (type(video), video)
        assert isinstance(representations, (list, tuple, RepresentationsList)), type(representations)
        self.video = video
        self.representations = RepresentationsList(representations)

    def set_compute_params(self, **kwargs) -> VideoRepresentationsExtractor:
        """Set the required params for all representations"""
        for r in self.representations:
            r.set_compute_params(**kwargs)
        return self

    def set_io_parameters(self, **kwargs) -> VideoRepresentationsExtractor:
        """Set the required params for all representations of IORepresentationMixin type"""
        for r in [_r for _r in self.representations if isinstance(_r, IORepresentationMixin)]:
            r.set_io_params(**kwargs)
        return self

    def run(self, output_dir: Path, frames: list[int] | None = None, output_dir_exists_mode: str = "raise",
            exception_mode: str = "stop_execution", n_threads_data_storer: int = 0,
            subset_exported_representations: list[str] | None = None) -> RunMetadata:
        """
        The main loop of the VRE. This will run all the representations on the video and store results in the output_dir
        Parameters:
        - output_dir The directory where VRE will store the representations
        - frames The list of frames to run VRE for. If None, will export all the frames of the video
        - output_dir_exists_mode What to do if the output dir already exists. See DataWriter docstring. Default 'raise'.
        - exception_mode What to do when encountering an exception. It always writes the exception to 'exception.txt'
          - 'skip_representation' Will stop the run of the current representation and start the next one
          - 'stop_execution' (default) Will stop the execution of VRE
        - n_threads_data_storer The number of threads used by the DataStorer
        - subset_exported_representations If set, only this subset of representations are exported, otherwise all of
            them based on these provided to the VRE constructor.
        Returns:
        - A RunMetadata object representing the run statistics for all representations of this run.
        """
        self._setup_logger(logs_dir := output_dir / ".logs", run_id := random_chars(n=10), now := now_fmt())
        self._setup_graphviz(logs_dir, run_id, now)
        if (subset := subset_exported_representations) is not None:
            logger.info(f"Explicit subset provided: {subset}. Exporting only these.")

        exported_reprs = self.representations.get_output_representations(subset=subset)
        assert len(exported_reprs) > 0, f"No output reprs returned, set I/O! {self.representations=}, {subset=}"
        runtime_args = VRERuntimeArgs(video=self.video, representations=exported_reprs, frames=frames,
                                      exception_mode=exception_mode, n_threads_data_storer=n_threads_data_storer)
        run_metadata = RunMetadata(repr_names=exported_reprs.names, runtime_args=runtime_args,
                                   logs_dir=logs_dir, now_str=now, run_id=run_id)
        logger.info(runtime_args)
        summary_printer = SummaryPrinter(exported_reprs.names, runtime_args)

        for vrepr in exported_reprs:
            dw = DataWriter(output_dir=output_dir, representation=vrepr, output_dir_exists_mode=output_dir_exists_mode)
            run_metadata.data_writers[vrepr.name] = dw.to_dict()
            repr_metadata = self.do_one_representation(run_id=run_metadata.id, representation=vrepr,
                                                       output_dir=output_dir,
                                                       output_dir_exists_mode=output_dir_exists_mode,
                                                       runtime_args=runtime_args)
            if repr_metadata.run_had_exceptions and runtime_args.exception_mode == "stop_execution":
                raise RuntimeError(f"Representation '{vrepr.name}' threw. "
                                   f"Check '{logger.get_file_handler().baseFilename}' for information")
            run_metadata.add_run_stats(repr_metadata)
            summary_printer.repr_metadatas[vrepr.name] = repr_metadata
        print(summary_printer())
        return run_metadata

    def do_one_representation(self, run_id: str, representation: Representation, output_dir: Path,
                              output_dir_exists_mode: str, runtime_args: VRERuntimeArgs) -> RepresentationMetadata:
        """The loop of each representation. Returns a representation metadata with information about this repr's run"""
        data_writer = DataWriter(output_dir=output_dir, representation=representation,
                                 output_dir_exists_mode=output_dir_exists_mode)
        data_storer = DataStorer(data_writer=data_writer, n_threads=runtime_args.n_threads_data_storer)
        logger.debug(f"Running {run_id=}:\n{representation}\n{data_storer}")
        rep: Representation | IORepresentationMixin = representation
        formats = sorted([f for f in [rep.image_format.value, rep.binary_format.value] if f != "not-set"])
        repr_metadata = RepresentationMetadata(repr_name=representation.name, formats=formats,
                                               disk_location=data_writer.rep_out_dir / ".repr_metadata.json",
                                               frames=list(range(len(self.video))))

        relevant_frames = [f for f in runtime_args.frames if f not in repr_metadata.frames_computed()]
        logger.debug(f"Out of {len(runtime_args.frames)} total frames, "
                     f"{len(runtime_args.frames) - len(relevant_frames)} are precomputed and will be skipped.")
        batches = make_batches(relevant_frames, rep.batch_size)

        pbar = tqdm(total=len(relevant_frames), desc=f"[VRE] {rep.name} bs={rep.batch_size}",
                    disable=os.getenv("VRE_PBAR", "1") == "0")
        for batch in batches:
            now = datetime.now()
            if data_writer.all_batch_exists(batch):
                pbar.update(len(batch))
                continue

            try:
                tr.cuda.empty_cache() # might empty some unused memory, not 100% if needed.
                rep_data = self._compute_one_representation_batch(rep, batch=batch, output_dir=data_writer.output_dir)
                if not rep.is_classification and rep.name.find("fastsam") == -1: # TODO: MemoryData of FSAM is binary...
                    assert rep_data.output.shape[-1] == rep.n_channels, (rep, rep_data.output, rep.n_channels)
                rep_data.output_images = rep.make_images(rep_data) if rep.export_image else None
                data_storer(rep_data)
            except Exception:
                self._log_error(f"\n[{rep.name} {rep.batch_size=} {batch=}] {traceback.format_exc()}\n")
                repr_metadata.add_time(None, batch, run_id=run_id, sync=True)
                repr_metadata.run_had_exceptions = True
                break
            repr_metadata.add_time((datetime.now() - now).total_seconds(), batch, run_id=run_id, sync=True)
            pbar.update(len(batch))

        # various cleanup stuff before ending with this representation
        data_storer.join_with_timeout(timeout=30)
        if isinstance(representation, LearnedRepresentationMixin) and representation.setup_called:
            representation.vre_free()
        for dep in representation.dependencies:
            if isinstance(dep, LearnedRepresentationMixin) and dep.setup_called:
                dep.vre_free()
        repr_metadata.store_on_disk()
        return repr_metadata

    # Helper private methods

    def _load_from_disk_if_possible(self, rep: Representation, video: VREVideo, ixs: list[int],
                                    output_dir: Path) -> ReprOut | None:
        """loads (batched) data from disk if possible."""
        assert isinstance(rep, IORepresentationMixin), rep
        assert isinstance(ixs, list) and all(isinstance(ix, int) for ix in ixs), (type(ixs), [type(ix) for ix in ixs])
        assert output_dir is not None and output_dir.exists(), output_dir
        # TODO: use data writer to create the paths as they can be non-npz for example
        npz_paths: list[Path] = [output_dir / rep.name / f"npz/{ix}.npz" for ix in ixs]
        extra_paths: list[Path] = [output_dir / rep.name / f"npz/{ix}_extra.npz" for ix in ixs]
        if any(not x.exists() for x in npz_paths): # partial batches are considered 'not existing' and overwritten
            return None
        extras_exist = [x.exists() for x in extra_paths]
        assert (ee := sum(extras_exist)) in (0, (ep := len(extra_paths))), f"Found {ee}. Expected either 0 or {ep}"
        disk_data: DiskData = np.array([rep.load_from_disk(x) for x in npz_paths])
        extra = [np.load(x, allow_pickle=True)["arr_0"].item() for x in extra_paths] if ee == ep else None
        logger.debug2(f"[{rep}] Slice: [{ixs[0]}:{ixs[-1]}]. All data found on disk and loaded")
        return ReprOut(frames=video[ixs], output=rep.disk_to_memory_fmt(disk_data), extra=extra, key=ixs)

    def _compute_one_representation_batch(self, rep: Representation, batch: list[int],
                                          output_dir: Path, depth: int = 0) -> ReprOut:
        # setting depth to higher values allows to compute deps of deps in memory. We throw to not hide bugs.
        # useful for VRE hacking (i.e. see semantic_mapper.py)
        assert depth <= 1, f"{rep=} {depth=} {batch=} {output_dir=}"
        loaded_data = self._load_from_disk_if_possible(rep, self.video, batch, output_dir)
        if loaded_data is not None:
            return loaded_data
        dep_data = []
        for dep in rep.dependencies:
            dep_data.append(self._compute_one_representation_batch(dep, batch, output_dir, depth + 1))
        if isinstance(rep, LearnedRepresentationMixin) and rep.setup_called is False:
            rep.vre_setup() # instantiates the model, loads to cuda device etc.
        return rep.compute(self.video, ixs=batch, dep_data=dep_data)

    def _setup_graphviz(self, logs_dir: Path, run_id: str, now_str: str):
        try:
            self.representations.to_graphviz().render(pth := f"{logs_dir}/graph-{run_id}-{now_str}",
                                                      format="png", cleanup=True)
            logger.info(f"Stored graphviz representation at: '{pth}.png'")
        except Exception as e:
            logger.error(e)

    def _setup_logger(self, logs_dir: Path, run_id: str, now_str: str):
        logs_dir.mkdir(exist_ok=True, parents=True)
        logs_file = logs_dir / f"logs-{run_id}-{now_str}.txt"
        logger.add_file_handler(logs_file)
        logger.info(f"Logging run at: '{logs_file}")

    def _log_error(self, msg: str):
        logs_file = logger.get_file_handler().baseFilename
        open(logs_file, "a").write(f"{'=' * 80}\n{now_fmt()}\n{msg}\n{'=' * 80}")
        logger.debug(f"Error: {msg}")

    def __call__(self, *args, **kwargs) -> RunMetadata:
        return self.run(*args, **kwargs)

    def __str__(self) -> str:
        return f"""\n[VRE]
- Video: {self.video}
- Representations: {self.representations}"""

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, ix: int | slice | range | list[int]) -> dict[str, ReprOut]:
        # Note: setup and free should be called outside of this function!
        if isinstance(ix, int):
            return self[[ix]]
        if isinstance(ix, slice):
            return self[list(range(ix.start, ix.stop))]
        if isinstance(ix, range):
            return self[list(ix)]
        assert isinstance(ix, list), type(ix)
        assert all(isinstance(r, IORepresentationMixin) for r in self.representations), self.representations

        res: dict[str, ReprOut] = {}
        for vre_repr in (pbar := tqdm(self.representations, disable=os.getenv("VRE_PBAR", "1") == "0")):
            pbar.set_description(f"[VRE Streaming] {vre_repr.name}")
            dep_names = [r.name for r in vre_repr.dependencies]
            batches = make_batches(ix, vre_repr.batch_size)
            batch_res: list[ReprOut] = []
            for b_ixs in batches:
                repr_out = vre_repr.compute(video=self.video, ixs=b_ixs,
                                            dep_data=[res[dep_name] for dep_name in dep_names])
                if vre_repr.output_size is not None:
                    repr_out = vre_repr.resize(repr_out, vre_repr.output_size)
                batch_res.append(repr_out)
            combined = ReprOut(frames=np.concatenate([br.frames for br in batch_res]),
                               key=sum([br.key for br in batch_res], []),
                               output=MemoryData(np.concatenate([br.output for br in batch_res])))
            if vre_repr.image_format.value != "not-set":
                combined.output_images = vre_repr.make_images(combined)
            res[vre_repr.name] = combined
        return res
