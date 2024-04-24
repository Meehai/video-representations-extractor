"""Video Representations Extractor module"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Any
from functools import partial, reduce
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

def _took(now: datetime.date, l: int, r: int) -> list[float]:
    return [(datetime.now() - now).total_seconds() / (r - l)] * (r - l)

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
        logger.info(
            f"""
  - Video path: '{self.video.file}'
  - Output dir: '{output_dir}'
  - Representations ({len(self.representations)}): {", ".join(x for x in self.representations.keys())}
  - Video shape: {self.video.shape}
  - Output frames ({end_frame - start_frame}): [{start_frame} : {end_frame - 1}]
  - Batch size: {batch_size}
  - Export npy: {export_npy}
  - Export png: {export_png}
"""
        )

    def _check_call_args(self, output_dir: Path, start_frame: int, end_frame: int, batch_size: int, export_npy: bool,
                         export_png: bool, output_dir_exist_mode: str,
                         representations_setup: dict[str, dict[str, Any]]):
        """check the args of the call method"""
        assert batch_size >= 1, f"batch size must be >= 1, got {batch_size}"
        assert export_npy + export_png > 0, "At least one of export modes must be True"
        assert isinstance(start_frame, int) and start_frame <= end_frame, (start_frame, end_frame)
        assert output_dir_exist_mode in ("overwrite", "skip_computed", "raise"), output_dir_exist_mode
        if output_dir.exists():
            valid = output_dir_exist_mode in ("overwrite", "skip_computed")
            assert valid, (f"'{output_dir}' exists. Set 'output_dir_exist_mode' to 'overwrite' or 'skip_computed'")
            if output_dir_exist_mode == "overwrite":
                logger.warning(f"Output dir '{output_dir}' already exists, will overwrite it")
                shutil.rmtree(output_dir)
        for name in representations_setup.keys():
            assert name in self.representations.keys(), f"Representation '{name}' not found in {self.representations}"

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

    def _do_one_representation(self, representation: Representation, batches: np.ndarray,
                               npy_paths: dict[str, list[Path]], png_paths: dict[str, list[Path]],
                               export_npy: bool, export_png: bool,
                               representations_setup: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
        repr_setup = representations_setup[representation.name]
        repr_png_paths = png_paths[representation.name]
        repr_npy_paths = npy_paths[representation.name]
        try:
            representation.vre_setup(video=self.video, **repr_setup)
        except TypeError as e:
            logger.error(f"{representation} => {repr_setup}")
            raise e

        left, right = batches[0:-1], batches[1:]
        start_frame, end_frame = left[0], right[-1]
        repr_stats = []
        pbar = tqdm(total=end_frame - start_frame, desc=f"[VRE] {representation.name}")
        for l, r in zip(left, right):
            now = datetime.now()
            (raw_data, extra), imgs = representation.vre_make(self.video, slice(l, r), export_png)
            self._store_data(raw_data, extra, imgs, repr_npy_paths, repr_png_paths, l, r, export_npy, export_png)
            # update the statistics and the progress bar
            repr_stats.extend(_took(now, l, r))
            pbar.update(r - l)
        pbar.close()
        return {representation.name: repr_stats}

    def run(self, output_dir: Path, export_npy: bool, export_png: bool, start_frame: int | None = None,
            end_frame: int | None = None, batch_size: int = 1, output_dir_exist_mode: str = "raise",
            representations_setup: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
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
        - representations_setup A dict of {representation_name: {representation_setup}}. This is used to pass
            representation specific inference parameters to the VRE, like setting device before running or loading
            some non-standard weights.
        Returns:
        - A dataframe with the run statistics for each representation
        """
        if representations_setup is None:
            logger.warning("representations_setup is None, default to empty dict")
            representations_setup = {r: {} for r in self.representations.keys()}
        if end_frame is None:
            end_frame = len(self.video)
            logger.warning(f"end frame not set, default to the last frame of the video: {len(self.video)}")
        if start_frame is None:
            start_frame = 0
            logger.warning("start frame not set, default to 0")
        if batch_size > end_frame - start_frame:
            logger.warning(f"batch size {batch_size} is larger than #frames to process [{start_frame}:{end_frame}].")
            batch_size = end_frame - start_frame
        self._check_call_args(output_dir, start_frame, end_frame, batch_size, export_npy,
                              export_png, output_dir_exist_mode, representations_setup)

        # run_stats will hold a dict: {repr_name: [time_taken, ...]} for all representations, for debugging/logging
        run_stats: dict[str, list[float]] = {}
        npy_paths, png_paths = self._make_run_paths(output_dir, export_npy, export_png)
        self._print_call(output_dir, start_frame, end_frame, batch_size, export_npy, export_png)

        last_one = min(end_frame, len(self.video))
        batches = np.arange(start_frame, last_one, batch_size)
        batches = np.array([*batches, last_one], dtype=np.int64) if batches[-1] != last_one else batches
        repr_fn = partial(self._do_one_representation, batches=batches,
                          npy_paths=npy_paths, png_paths=png_paths, export_npy=export_npy,
                          export_png=export_png, representations_setup=representations_setup)
        run_stats = list(map(repr_fn, self.representations.values()))

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
