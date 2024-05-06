"""Helper module to make sense of the arguments sent to vre.run()"""
from typing import Any
from pathlib import Path
import shutil

from .utils import is_dir_empty
from .logger import logger

RepresentationsSetup = dict[str, dict[str, Any]]

class VRERuntimeArgs:
    """VRE runtime args. Helper class to process the arguments sent to vre.run()"""
    def __init__(self, vre: "VRE", start_frame: int | None, end_frame: int | None, batch_size: int, export_npy: bool,
                 export_png: bool, reprs_setup: RepresentationsSetup | None, output_dir_exist_mode: str,
                 exception_mode: str):
        assert batch_size >= 1, f"batch size must be >= 1, got {batch_size}"
        assert export_npy + export_png > 0, "At least one of export modes must be True"
        assert output_dir_exist_mode in ("overwrite", "skip_computed", "raise"), output_dir_exist_mode
        assert exception_mode in ("stop_execution", "skip_representation"), exception_mode
        if reprs_setup is None:
            logger.warning("reprs_setup is None, default to empty dict")
            reprs_setup = {r: {} for r in vre.representations.keys()}
        if end_frame is None:
            end_frame = len(vre.video)
            logger.warning(f"end frame not set, default to the last frame of the video: {len(vre.video)}")
        if start_frame is None:
            start_frame = 0
            logger.warning("start frame not set, default to 0")

        assert isinstance(start_frame, int) and start_frame <= end_frame, (start_frame, end_frame)
        for name in reprs_setup.keys():
            assert name in vre.representations.keys(), f"Representation '{name}' not found in {vre.representations}"
        self.vre = vre
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.batch_size = batch_size
        self.export_npy = export_npy
        self.export_png = export_png
        self.reprs_setup = reprs_setup
        self.output_dir_exist_mode = output_dir_exist_mode
        self.exception_mode = exception_mode

        self.batch_sizes = {}
        for r in vre.representations.values():
            if hasattr(r, "batch_size"): # in case it's provided in cfg (TODO: make this nicer w/o getattr)
                logger.info(f"Representation '{r}' has explicit batch size: {r.batch_size}")
            self.batch_sizes[r.name] = min(getattr(r, "batch_size", batch_size), batch_size)
        self.npy_paths: dict[str, list[Path]] = self._make_npy_paths() # {repr: [out_dir/npy/0.npz, ...]}
        self.png_paths: dict[str, list[Path]] = self._make_png_paths() # {repr: [out_dir/png/0.png, ...]}
        assert len(self.npy_paths) > 0 and len(self.png_paths) > 0, (len(self.npy_paths), len(self.png_paths))
        self._make_and_check_dirs()
        self._print_call()

    def _make_npy_paths(self) -> dict[str, list[Path]]:
        npy_paths = {}
        for name in self.vre.representations.keys():
            npy_base_dir = self.vre.output_dir / name / "npy/"
            npy_paths[name] = [npy_base_dir / f"{t}.npz" for t in range(len(self.vre.video))]
        return npy_paths

    def _make_png_paths(self) -> dict[str, list[Path]]:
        png_paths = {}
        for name in self.vre.representations.keys():
            png_base_dir = self.vre.output_dir / name / "png/"
            png_paths[name] = [png_base_dir / f"{t}.png" for t in range(len(self.vre.video))]
        return png_paths

    def _print_call(self):
        logger.info(f"""
  - Video path: '{self.vre.video.file}'
  - Output dir: '{self.vre.output_dir}' (exist mode: '{self.output_dir_exist_mode}')
  - Representations ({len(self.vre.representations)}): {", ".join(x for x in self.vre.representations.keys())}
  - Video shape: {self.vre.video.shape} (FPS: {self.vre.video.frame_rate})
  - Output frames ({self.end_frame - self.start_frame}): [{self.start_frame} : {self.end_frame - 1}]
  - Batch size: {self.batch_size}
  - Export npy: {self.export_npy}
  - Export png: {self.export_png}
  - Exception mode: '{self.exception_mode}'
""")

    def _make_and_check_dirs(self):
        for representation in self.vre.representations:
            npy_dir: Path = self.npy_paths[representation][0].parent
            png_dir: Path = self.png_paths[representation][0].parent

            if self.export_npy and npy_dir.exists() and not is_dir_empty(npy_dir, "*.npz"):
                if self.output_dir_exist_mode == "overwrite":
                    logger.debug(f"Output dir '{npy_dir}' already exists, will overwrite it")
                    shutil.rmtree(npy_dir)
                else:
                    assert self.output_dir_exist_mode != "raise", \
                        f"'{npy_dir}' exists. Set --output_dir_exist_mode to 'overwrite' or 'skip_computed'"
            npy_dir.mkdir(exist_ok=True, parents=True)

            if self.export_png and png_dir.exists() and not is_dir_empty(png_dir, "*.png"):
                if self.output_dir_exist_mode == "overwrite":
                    logger.debug(f"Output dir '{png_dir}' already exists, will overwrite it")
                    shutil.rmtree(png_dir)
                else:
                    assert self.output_dir_exist_mode != "raise", \
                        f"'{png_dir}' exists. Set --output_dir_exist_mode to 'overwrite' or 'skip_computed'"
            png_dir.mkdir(exist_ok=True, parents=True)
