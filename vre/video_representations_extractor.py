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
from .metadata import RunMetadata, RepresentationMetadata
from .vre_video import VREVideo
from .utils import now_fmt, make_batches, vre_topo_sort, ReprOut, DiskData, SummaryPrinter
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
            exported_representations: list[str] | None = None) -> RunMetadata:
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
        - exported_representations If set, only this subset of representations are exported, otherwise all of them
            based on these provided to the VRE constructor.
        Returns:
        - A RunMetadata object representing the run statistics for all representations of this run.
        """
        self._setup_logger(output_dir, now := now_fmt())
        try:
            self.to_graphviz().render(pth := f"{output_dir}/.logs/graph-{now}", format="png", cleanup=True)
            logger.info(f"Stored graphviz representation at: '{pth}.png'")
        except Exception as e:
            logger.error(e)

        runtime_args = VRERuntimeArgs(self.video, self.representations, frames, exception_mode, n_threads_data_storer)
        logger.info(runtime_args)
        run_metadata = RunMetadata(self.repr_names, runtime_args, output_dir / f".logs/run_metadata-{now}.json")
        summary_printer = SummaryPrinter(self.repr_names, runtime_args)
        for vrepr in self.representations:
            data_writer = DataWriter(output_dir=output_dir, representation=vrepr,
                                     output_dir_exists_mode=output_dir_exists_mode)
            run_metadata.data_writers[vrepr.name] = data_writer.to_dict()
        run_metadata.store_on_disk()

        for vre_repr in self._get_output_representations(exported_representations): # checks are done inside get fn
            repr_metadata = self._do_one_representation(representation=vre_repr, output_dir=output_dir,
                                                        output_dir_exists_mode=output_dir_exists_mode,
                                                        runtime_args=runtime_args,)
            if repr_metadata.run_had_exceptions and runtime_args.exception_mode == "stop_execution":
                raise RuntimeError(f"Representation '{vre_repr.name}' threw. "
                                   f"Check '{logger.get_file_handler().baseFilename}' for information")
            summary_printer.repr_metadatas[vre_repr.name] = repr_metadata
        print(summary_printer())
        return run_metadata

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

    def _do_one_representation(self, representation: Representation, output_dir: Path, output_dir_exists_mode: str,
                               runtime_args: VRERuntimeArgs) -> RepresentationMetadata:
        """The loop of each representation. Returns a representation metadata with information about this repr's run"""
        data_writer = DataWriter(output_dir=output_dir, representation=representation,
                                 output_dir_exists_mode=output_dir_exists_mode)
        data_storer = DataStorer(data_writer=data_writer, n_threads=runtime_args.n_threads_data_storer)
        logger.info(f"Running:\n{representation}\n{data_storer}")
        rep: Representation | ComputeRepresentationMixin | IORepresentationMixin = representation
        repr_metadata = RepresentationMetadata(repr_name=representation.name,
                                               disk_location=data_writer.rep_out_dir / ".repr_metadata.json",
                                               frames=list(range(len(self.video))))
        rep.output_size = self.video.frame_shape[0:2] if rep.output_size == "video_shape" else rep.output_size

        relevant_frames = [f for f in runtime_args.frames if f not in map(int, repr_metadata.frames_computed)]
        logger.debug(f"Out of {len(runtime_args.frames)} total frames, "
                     f"{len(runtime_args.frames) - len(relevant_frames)} are precomputed and will be skipped.")
        batches = make_batches(relevant_frames, rep.batch_size)
        pbar = tqdm(total=len(relevant_frames), desc=f"[VRE] {rep.name} bs={rep.batch_size}")
        for batch in batches:
            now = datetime.now()
            if data_writer.all_batch_exists(batch):
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
                repr_metadata.add_time((1 << 31) * len(batch), batch)
                repr_metadata.run_had_exceptions = True
                break
            repr_metadata.add_time((datetime.now() - now).total_seconds(), batch)
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
    def _setup_logger(self, output_dir: Path, now_str: str):
        (logs_dir := output_dir / ".logs").mkdir(exist_ok=True, parents=True)
        logs_file = logs_dir / f"logs-{now_str}.txt"
        logger.add_file_handler(logs_file)
        logger.info(f"Logging run at: '{logs_file}")

    def _log_error(self, msg: str):
        logs_file = logger.get_file_handler().baseFilename
        open(logs_file, "a").write(f"{'=' * 80}\n{now_fmt()}\n{msg}\n{'=' * 80}")
        logger.debug(f"Error: {msg}")

    def _get_output_representations(self, exported_representations: list[str]) \
            -> list[IORepresentationMixin | Representation]:
        """given all the representations, keep those that actually export something. At least one is needed"""
        crs: list[IORepresentationMixin] = [_r for _r in self.representations if isinstance(_r, IORepresentationMixin)]
        assert len(crs) > 0, f"No I/O Representation found in {self.repr_names}"
        out_r: list[Representation] = [r for r in crs if r.export_binary or r.export_image]
        if exported_representations:
            logger.info(f"Explicit subset providewd: {exported_representations}. Exporting only these.")
            out_r = [r for r in out_r if r.name in exported_representations]

        assert len(out_r) > 0, f"No output format set for any I/O Representation: {', '.join([r.name for r in crs])}"
        for vre_repr in out_r:
            assert isinstance(vre_repr, Representation), vre_repr
            assert isinstance(vre_repr, IORepresentationMixin), vre_repr
            assert vre_repr.binary_format is not None or vre_repr.image_format is not None, vre_repr

        return out_r

    def __call__(self, *args, **kwargs) -> RunMetadata:
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
