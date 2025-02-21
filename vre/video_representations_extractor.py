"""Video Repentations Extractor module"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import traceback
from tqdm import tqdm
import torch as tr
import numpy as np

from .representations import Representation, ComputeRepresentationMixin, LearnedRepresentationMixin
from .representations.io_representation_mixin import IORepresentationMixin
from .vre_runtime_args import VRERuntimeArgs
from .data_writer import DataWriter
from .data_storer import DataStorer
from .metadata import Metadata
from .vre_video import VREVideo
from .utils import now_fmt, make_batches, vre_topo_sort, ReprOut, DiskData
from .logger import vre_logger as logger

class VideoRepresentationsExtractor:
    """Video Representations Extractor class"""

    def __init__(self, video: VREVideo, representations: list[Representation]):
        """
        Parameters:
        - video The video we are performing VRE one
        - representations The list instantiated representations. Must be topo-sortable based on name and deps.
        """
        assert len(representations) > 0, "At least one representation must be provided"
        assert all(isinstance(x, Representation) for x in representations), [(x.name, type(x)) for x in representations]
        assert isinstance(video, VREVideo), (type(video), video)
        self.video = video
        self.representations: list[Representation] = vre_topo_sort(representations)
        self.repr_names = [r.name for r in representations]
        self._logs_file: Path | None = None
        self._metadata: Metadata | None = None

    def set_compute_params(self, **kwargs) -> VideoRepresentationsExtractor:
        """Set the required params for all representations of ComputeRepresentationMixin type"""
        for r in [_r for _r in self.representations if isinstance(_r, ComputeRepresentationMixin)]:
            r.set_compute_params(**kwargs)
        return self

    def set_io_parameters(self, **kwargs) -> VideoRepresentationsExtractor:
        """Set the required params for all representations of IORepresentationMixin type"""
        for r in [_r for _r in self.representations if isinstance(_r, IORepresentationMixin)]:
            r.set_io_params(**kwargs)
        return self

    def to_graphviz(self, **kwargs) -> "Digraph":
        """Returns a graphviz object from this graph. Used for plotting the graph. Best for smaller graphs."""
        from graphviz import Digraph # pylint: disable=import-outside-toplevel
        g = Digraph()
        for k, v in kwargs.items():
            g.attr(**{k: v})
        g.attr(rankdir="LR")
        for node in self.representations:
            g.node(name=f"{node.name}", shape="oval")
        edges: list[tuple[str, str]] = [(r.name, dep.name) for r in self.representations for dep in r.dependencies]
        for l, r in edges:
            g.edge(r, l) # reverse?
        return g

    def run(self, output_dir: Path, frames: list[int] | None = None, output_dir_exists_mode: str = "raise",
            exception_mode: str = "stop_execution", n_threads_data_storer: int = 0,
            exported_representations: list[str] | None = None) -> Metadata:
        """
        The main loop of the VRE. This will run all the representations on the video and store results in the output_dir
        Parameters:
        - output_dir The directory where VRE will store the representations
        - frames The list of frames to run VRE for. If None, will export all the frames of the video
        - exception_mode What to do when encountering an exception. It always writes the exception to 'exception.txt'
          - 'skip_representation' Will stop the run of the current representation and start the next one
          - 'stop_execution' (default) Will stop the execution of VRE
        - n_threads_data_storer The number of threads used by the DataStorer
        - exported_representations If set, only this subset of representations are exported, otherwise all of them
            based on these provided to the VRE constructor.
        Returns:
        - A dataframe with the run statistics for each representation
        """
        self._setup_logger(output_dir, now := now_fmt())
        try:
            self.to_graphviz().render(pth := f"{output_dir}/.logs/graph-{now}", format="png", cleanup=True)
            logger.info(f"Stored graphviz representation at: '{pth}.png'")
        except ImportError:
            pass

        runtime_args = VRERuntimeArgs(self.video, self.representations, frames, exception_mode, n_threads_data_storer)
        logger.info(runtime_args)
        self._metadata = Metadata(self.repr_names, runtime_args, self._logs_file.parent / f"run_metadata-{now}.json")
        for vre_repr in self._get_output_representations(exported_representations):
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

    # Private methods related to the main computation of each representation
    def _cleanup_one_representation(self, representation: Representation, data_storer: DataStorer):
        data_storer.join_with_timeout(timeout=30)
        if isinstance(representation, LearnedRepresentationMixin) and representation.setup_called:
            representation.vre_free()
        for dep in representation.dependencies:
            if isinstance(dep, LearnedRepresentationMixin) and dep.setup_called:
                dep.vre_free()

    def _load_from_disk_if_possible(self, rep: Representation, video: VREVideo, ixs: list[int], output_dir: Path):
        """loads (batched) data from disk if possible."""
        assert isinstance(rep, IORepresentationMixin), rep
        assert isinstance(ixs, list) and all(isinstance(ix, int) for ix in ixs), (type(ixs), [type(ix) for ix in ixs])
        assert output_dir is not None and output_dir.exists(), output_dir
        # TODO: use data writer to create the paths as they can be non-npz for example
        npz_paths: list[Path] = [output_dir / rep.name / f"npz/{ix}.npz" for ix in ixs]
        extra_paths: list[Path] = [output_dir / rep.name / f"npz/{ix}_extra.npz" for ix in ixs]
        if any(not x.exists() for x in npz_paths): # partial batches are considered 'not existing' and overwritten
            return
        extras_exist = [x.exists() for x in extra_paths]
        assert (ee := sum(extras_exist)) in (0, (ep := len(extra_paths))), f"Found {ee}. Expected either 0 or {ep}"
        disk_data: DiskData = np.array([rep.load_from_disk(x) for x in npz_paths])
        extra = [np.load(x, allow_pickle=True)["arr_0"].item() for x in extra_paths] if ee == ep else None
        logger.debug2(f"[{rep}] Slice: [{ixs[0]}:{ixs[-1]}]. All data found on disk and loaded")
        rep.data = ReprOut(frames=video[ixs], output=rep.disk_to_memory_fmt(disk_data), extra=extra, key=ixs)

    def _compute_one_representation_batch(self, rep: Representation, batch: list[int],
                                          output_dir: Path, depth: int = 0):
        assert isinstance(rep, ComputeRepresentationMixin), rep
        # setting depth to higher values allows to compute deps of deps in memory. We throw to not hide bugs.
        # useful for VRE hacking (i.e. see semanit_mapper.py)
        assert depth <= 1, f"{rep=} {depth=} {batch=} {output_dir=}"
        # TODO: make unit test with this (lingering .data from previous computation) on depth == 1 (i.e. deps of rep)
        rep.data = None # Important to invalidate any previous results here
        self._load_from_disk_if_possible(rep, self.video, batch, output_dir)
        if rep.data is not None:
            return
        for dep in rep.dependencies:
            self._compute_one_representation_batch(dep, batch, output_dir, depth + 1)
        if isinstance(rep, LearnedRepresentationMixin) and rep.setup_called is False:
            rep.vre_setup() # instantiates the model, loads to cuda device etc.
        rep.compute(self.video, ixs=batch)
        assert rep.data is not None, f"{rep=} {batch=} {output_dir=} {depth=}"

    def _do_one_representation(self, data_writer: DataWriter, runtime_args: VRERuntimeArgs) -> bool:
        """main loop of each representation. Returns true if had exception, false otherwise"""
        data_storer = DataStorer(data_writer, runtime_args.n_threads_data_storer)
        logger.info(f"Running:\n{data_writer.rep}\n{data_storer}")
        rep: Representation | ComputeRepresentationMixin | IORepresentationMixin = data_writer.rep
        self._metadata.metadata["data_writers"][rep.name] = data_writer.to_dict()
        rep.output_size = self.video.frame_shape[0:2] if rep.output_size == "video_shape" else rep.output_size

        batches = make_batches(runtime_args.frames, rep.batch_size)
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
                if not rep.is_classification and rep.name.find("fastsam") == -1: # TODO: MemoryData of M2F is binary...
                    assert rep.data.output.shape[-1] == rep.n_channels, (rep, rep.data.output, rep.n_channels)
                rep.data.output_images = rep.make_images(rep.data) if rep.export_image else None
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

    # Helper private methods
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

    def _get_output_representations(self, exported_representations: list[str]) -> list[IORepresentationMixin]:
        """given all the representations, keep those that actually export something. At least one is needed"""
        crs: list[IORepresentationMixin] = [_r for _r in self.representations if isinstance(_r, IORepresentationMixin)]
        assert len(crs) > 0, f"No I/O Representation found in {self.repr_names}"
        out_r: list[Representation] = [r for r in crs if r.export_binary or r.export_image]
        if exported_representations:
            logger.info(f"Explicit subset providewd: {exported_representations}. Exporting only these.")
            out_r = [r for r in out_r if r.name in exported_representations]
        assert len(out_r) > 0, f"No output format set for any I/O Representation: {', '.join([r.name for r in crs])}"
        return out_r

    def __call__(self, *args, **kwargs) -> Metadata:
        return self.run(*args, **kwargs)

    def __str__(self) -> str:
        return f"""\n[VRE]
- Video: {self.video}
- Representations ({len(self.representations)}): [{", ".join(self.repr_names)}]"""

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key: str) -> Representation:
        assert key in self.repr_names, f"Representation '{key}' not in {self.repr_names}"
        return self.representations[self.repr_names.index(key)]
