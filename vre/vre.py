"""Video Representations Extractor module"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Any
from functools import reduce
import shutil
from tqdm import tqdm
from omegaconf import DictConfig
import pims
import numpy as np
import pandas as pd

from .representation import Representation
from .logger import logger
from .utils import image_write, FakeVideo

RunPaths = tuple[dict[str, list[Path]], dict[str, list[Path]]]
RepresentationsSetup = dict[str, dict[str, Any]]

def _took(now: datetime.date, l: int, r: int) -> list[float]:
    return [(datetime.now() - now).total_seconds() / (r - l)] * (r - l)

def _make_batches(video: pims.Video, start_frame: int, end_frame: int, batch_size: int) -> np.ndarray:
    """return 1D array [start_frame, start_frame+bs, start_frame+2*bs... end_frame]"""
    if batch_size > end_frame - start_frame:
        logger.warning(f"batch size {batch_size} is larger than #frames to process [{start_frame}:{end_frame}].")
        batch_size = end_frame - start_frame
    last_one = min(end_frame, len(video))
    batches = np.arange(start_frame, last_one, batch_size)
    batches = np.array([*batches, last_one], dtype=np.int64) if batches[-1] != last_one else batches
    return batches

class VRE:
    """Video Representations Extractor class"""

    def __init__(self, video: pims.Video, representations: dict[str, Representation | type[Representation]]):
        """
        Parameters:
        - video The video we are performing VRE one
        - representations The dict of instantiated and topo sorted representations (or callable to instantiate them)
        """
        assert len(representations) > 0, "At least one representation must be provided"
        assert isinstance(video, (pims.Video, FakeVideo)), type(video)
        self.video = video
        self.representations: dict[str, Representation] = representations

    def _make_run_paths(self, output_dir: Path, export_npy: bool, export_png: bool) -> RunPaths:
        """Create the output dirs structure. We may have 2 types of outputs: npy and png."""
        output_dir.mkdir(parents=True, exist_ok=True)

        npy_paths, png_paths = {}, {}
        for name in self.representations.keys():
            (output_dir / name).mkdir(exist_ok=True, parents=True)
            npy_base_dir = output_dir / name / "npy/"
            png_base_dir = output_dir / name / "png/"

            npy_paths[name] = [npy_base_dir / f"{t}.npz" for t in range(len(self.video))]
            png_paths[name] = [png_base_dir / f"{t}.png" for t in range(len(self.video))]

            if export_npy:
                npy_base_dir.mkdir(exist_ok=True)
            if export_png:
                png_base_dir.mkdir(exist_ok=True, parents=True)
        return npy_paths, png_paths

    def run_cfg(self, output_dir: Path, cfg: DictConfig):
        """runs VRE given a config file. This is testing the real case of using a vre config file when running"""
        start_frame = int(cfg.get("start_frame")) if cfg.get("start_frame") is not None else 0
        end_frame = int(cfg.get("end_frame") if cfg.get("end_frame") is not None else len(self.video))
        return self(output_dir=output_dir, start_frame=start_frame, end_frame=end_frame,
                    export_npy=cfg.get("export_npy"), export_png=cfg.get("export_png"),
                    batch_size=cfg.get("batch_size", 1),
                    output_dir_exist_mode=cfg.get("output_dir_exist_mode", "raise"))

    def _print_call(self, output_dir: Path, start_frame: int, end_frame: int, batch_size: int,
                    export_npy: bool, export_png: bool):
        logger.info(f"""
  - Video path: '{self.video.file}'
  - Output dir: '{output_dir}'
  - Representations ({len(self.representations)}): {", ".join(x for x in self.representations.keys())}
  - Video shape: {self.video.shape}
  - Output frames ({end_frame - start_frame}): [{start_frame} : {end_frame - 1}]
  - Batch size: {batch_size}
  - Export npy: {export_npy}
  - Export png: {export_png}
""")

    def _check_call_args(self, output_dir: Path, start_frame: int | None, end_frame: int | None, batch_size: int,
                         export_npy: bool, export_png: bool, output_dir_exist_mode: str,
                         reprs_setup: RepresentationsSetup | None) -> tuple[int, int, RepresentationsSetup]:
        """check the args of the call method"""
        assert batch_size >= 1, f"batch size must be >= 1, got {batch_size}"
        assert export_npy + export_png > 0, "At least one of export modes must be True"

        if reprs_setup is None:
            logger.warning("reprs_setup is None, default to empty dict")
            reprs_setup = {r: {} for r in self.representations.keys()}
        if end_frame is None:
            end_frame = len(self.video)
            logger.warning(f"end frame not set, default to the last frame of the video: {len(self.video)}")
        if start_frame is None:
            start_frame = 0
            logger.warning("start frame not set, default to 0")

        assert isinstance(start_frame, int) and start_frame <= end_frame, (start_frame, end_frame)
        assert output_dir_exist_mode in ("overwrite", "skip_computed", "raise"), output_dir_exist_mode
        for name in reprs_setup.keys():
            assert name in self.representations.keys(), f"Representation '{name}' not found in {self.representations}"
            if (p := output_dir / name).exists():
                if output_dir_exist_mode == "overwrite":
                    logger.debug(f"Output dir '{p}' already exists, will overwrite it")
                    shutil.rmtree(p)
                else:
                    assert output_dir_exist_mode != "raise", f"'{p}' exists. Set mode to 'overwrite' or 'skip_computed'"
        return start_frame, end_frame, reprs_setup

    def _store_data(self, raw_data: np.ndarray, extra: dict, imgs: np.ndarray | None, npy_paths: list[Path],
                    png_paths: list[Path], l: int, r: int, export_npy: bool, export_png: bool):
        """store the data in the right format"""
        h, w = self.video.frame_shape[0:2]
        if export_png:
            assert imgs is not None
            assert imgs.shape == (r - l, h, w, 3), (imgs.shape, (r - l, h, w, 3))
            assert imgs.dtype == np.uint8, imgs.dtype

        for i, t in enumerate(range(l, r)):
            if export_npy:
                if not npy_paths[t].exists():
                    np.savez(npy_paths[t], raw_data[i])
                    if len(extra) > 0:
                        np.savez(npy_paths[t].parent / f"{t}_extra.npz", extra[i])
            if export_png:
                image_write(imgs[i], png_paths[t])

    def _all_batch_exists(self, npy_paths: list[Path], png_paths: list[Path], l: int, r: int,
                          export_npy: bool, export_png: bool) -> bool:
        for ix in range(l, r):
            if export_npy and not npy_paths[ix].exists():
                return False
            if export_png and not png_paths[ix].exists():
                return False
        logger.debug(f"Batch [{l}:{r}] skipped.")
        return True

    def _do_one_representation(self, representation: Representation, start_frame: int, end_frame: int, batch_size: int,
                               npy_paths: dict[str, list[Path]], png_paths: dict[str, list[Path]], export_npy: bool,
                               export_png: bool, repr_setup: RepresentationsSetup) -> dict[str, list[float]]:
        """main loop of each representation."""
        name = representation.name
        batch_size = min(getattr(representation, "batch_size", batch_size), batch_size) # in case it's provided in cfg
        # call vre_setup here so expensive representations get lazy deep instantiated (i.e. models loading)
        try:
            representation.vre_setup(video=self.video, **repr_setup)
        except Exception as e:
            open("exception.txt", "a").write(f"\n[{name} {datetime.now()} {batch_size=} {e}\n")
            del representation
            return {name: [1 << 31] * (end_frame - start_frame)}

        batches = _make_batches(self.video, start_frame, end_frame, batch_size)
        left, right = batches[0:-1], batches[1:]
        repr_stats = []
        pbar = tqdm(total=end_frame - start_frame, desc=f"[VRE] {name} bs={batch_size}")
        for l, r in zip(left, right): # main VRE loop
            if self._all_batch_exists(npy_paths, png_paths, l, r, export_npy, export_png):
                pbar.update(r - l)
                repr_stats.extend(_took(datetime.now(), l, r))
                continue

            now = datetime.now()
            try:
                # TODO(!27): move this to VRE
                (raw_data, extra), imgs = representation.vre_make(self.video, slice(l, r), export_png) # noqa
                self._store_data(raw_data, extra, imgs, npy_paths, png_paths, l, r, export_npy, export_png)
            except Exception as e:
                open("exception.txt", "a").write(f"\n[{name} {now} {batch_size=} {l=} {r=}] {e}\n")
                repr_stats.extend([1 << 31] * (end_frame - l))
                del representation # noqa
                break
            # update the statistics and the progress bar
            repr_stats.extend(_took(now, l, r))
            pbar.update(r - l)
        return {name: repr_stats}

    def run(self, output_dir: Path, export_npy: bool, export_png: bool, start_frame: int | None = None,
            end_frame: int | None = None, batch_size: int | dict[str, int] = 1, output_dir_exist_mode: str = "raise",
            reprs_setup: RepresentationsSetup | None = None) -> pd.DataFrame:
        """
        The main loop of the VRE. This will run all the representations on the video and store results in the output_dir
        Parameteres:
        - output_dir The output directory where to store the results
        - export_npy Whether to export the npy files
        - export_png Whether to export the png files
        - start_frame The first frame to process (inclusive)
        - end_frame The last frame to process (inclusive)
        - batch_size The batch size to use when processing the video
        - output_dir_exist_mode What to do if the output dir already exists. Can be one of:
          - 'overwrite' Overwrite the output dir if it already exists
          - 'skip_computed' Skip the computed frames and continue from the last computed frame
          - 'raise' Raise an error if the output dir already exists
        - reprs_setup A dict of {representation_name: {representation_setup}}. This is used to pass
            representation specific inference parameters to the VRE, like setting device before running or loading
            some non-standard weights.
        Returns:
        - A dataframe with the run statistics for each representation
        """
        start_frame, end_frame, reprs_setup = self._check_call_args(output_dir, start_frame, end_frame, batch_size,
                                                                    export_npy, export_png, output_dir_exist_mode,
                                                                    reprs_setup)

        # run_stats will hold a dict: {repr_name: [time_taken, ...]} for all representations, for debugging/logging
        run_stats: dict[str, list[float]] = {}
        npy_paths, png_paths = self._make_run_paths(output_dir, export_npy, export_png)
        self._print_call(output_dir, start_frame, end_frame, batch_size, export_npy, export_png)

        run_stats = []
        for name, vre_repr in self.representations.items():
            repr_stats = self._do_one_representation(vre_repr, start_frame=start_frame, end_frame=end_frame,
                                                     batch_size=batch_size, npy_paths=npy_paths[name],
                                                     png_paths=png_paths[name], export_npy=export_npy,
                                                     export_png=export_npy, repr_setup=reprs_setup[name])
            run_stats.append(repr_stats)
        df_run_stats = pd.DataFrame(reduce(lambda a, b: {**a, **b}, run_stats), index=range(start_frame, end_frame))
        return df_run_stats

    # pylint: disable=too-many-branches, too-many-nested-blocks
    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.run(*args, **kwargs)

    def __str__(self) -> str:
        return (
            f"VRE ({len(self.representations)} representations). "
            f"Video: '{self.video.raw_data.path}' (shape: {self.video.shape})"
        )

    def __repr__(self) -> str:
        return str(self)
