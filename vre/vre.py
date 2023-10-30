"""Video Representations Extractor module"""
from __future__ import annotations
from pathlib import Path
from typing import Callable
from datetime import datetime
from tqdm import tqdm
import cv2
from omegaconf import DictConfig
import pims
import numpy as np
import pandas as pd

from .representation import Representation
from .logger import logger

class VRE:
    """Video Representations Extractor class"""

    def __init__(self, video: pims.Video, representations: dict[str, Representation | type[Representation]]):
        """
        Parameters:
        - video The video we are performing VRE one
        - representations The dict of instantiated and topo sorted representations (or callable to instantiate them)
        """
        assert len(representations) > 0
        assert isinstance(video, pims.Video), type(video)
        self.video = video
        self._tsr = representations

    @property
    def tsr(self) -> dict[str, Representation]:
        """Call the representations module to build each of them, one by one, after topo sorting for dependencies"""
        return self._tsr

    def _make_run_paths(self, output_dir: Path, output_resolution: tuple[int, int],
                        export_raw: bool, export_npy: bool, export_png: bool) -> tuple[dict, dict, dict]:
        """
        create the output dirs structure. We may have 2 types of outputs: npy and png for each, we may have
        different resolutions, so we store them under npy/HxW or png/HxW for the npy we always store the raw output
        under 'raw' from which we can derive all the others even at a later time
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_resolution_str = f"{output_resolution[0]}x{output_resolution[1]}"

        npy_raw_paths, npy_resz_paths, png_resz_paths = {}, {}, {}
        for name in self.tsr.keys():
            (output_dir / name).mkdir(exist_ok=True, parents=True)
            npy_raw_paths[name] = output_dir / name / "npy/raw"
            npy_resz_paths[name] = output_dir / name / f"npy/{out_resolution_str}"
            png_resz_paths[name] = output_dir / name / f"png/{out_resolution_str}"
            if export_raw:
                npy_raw_paths[name].mkdir(exist_ok=True, parents=True)
            if export_npy:
                npy_resz_paths[name].mkdir(exist_ok=True)
            if export_png:
                png_resz_paths[name].mkdir(exist_ok=True, parents=True)
        return npy_raw_paths, npy_resz_paths, png_resz_paths

    def run_cfg(self, output_dir: Path, cfg: DictConfig):
        """runs VRE given a config file. This is testing the real case of using a vre config file when running"""
        start_frame = int(cfg.get("start_frame")) if cfg.get("start_frame") is not None else 0
        end_frame = int(cfg.get("end_frame") if cfg.get("end_frame") is not None else len(self.video))
        return self(output_dir=output_dir, start_frame=start_frame, end_frame=end_frame,
                    export_raw=cfg.get("export_raw"), export_npy=cfg.get("export_npy"),
                    export_png=cfg.get("export_png"), output_resolution=cfg.get("output_resolution"))

    def _print(self, output_dir: Path, start_frame: int, end_frame: int, output_resolution: tuple[int, int],
               export_raw: bool, export_npy: bool, export_png: bool):
        logger.info(
            f"""
  - Video path: '{self.video.file}'
  - Output dir: '{output_dir}'
  - Representations ({len(self.tsr)}): {", ".join([x for x in self.tsr.keys()])}
  - Video shape: {self.video.shape}
  - Output frames ({end_frame - start_frame}): [{start_frame} : {end_frame - 1}]
  - Output resolution: {tuple(output_resolution)}
  - Export raw npy: {export_raw}
  - Export resized npy: {export_npy}
  - Export resized png: {export_png}
"""
        )

    def __call__(self, output_dir: Path, start_frame: int | None = None, end_frame: int | None = None,
                 batch_size: int = 1, output_resolution: tuple[int, int] | None = None,
                 export_raw: bool = False, export_npy: bool = False, export_png: bool = False,
                 ) -> pd.DataFrame:
        assert export_raw + export_png + export_npy > 0, "At least one of export modes must be True"
        if output_resolution is None:
            output_resolution = (self.video.frame_shape[0], self.video.frame_shape[1])
            logger.warning(f"output resolution not set, default to video shape: {output_resolution}")
        if end_frame is None:
            end_frame = len(self.video)
            logger.warning(f"end frame not set, default to the last frame of the video: {len(self.video)}")
        if start_frame is None:
            start_frame = 0
            logger.warning(f"start frame not set, default to 0")
        assert isinstance(start_frame, int) and start_frame <= end_frame, (start_frame, end_frame)
        # run_stats will hold a dict: {repr_name: [time_taken, ...]} for all representations, for debugging/logging
        run_stats = {repr_name: [] for repr_name in ["frame", *self.tsr.keys()]}

        npy_raw_paths, npy_resz_paths, png_resz_paths = self._make_run_paths(output_dir, output_resolution,
                                                                             export_raw, export_npy, export_png)
        self._print(output_dir, start_frame, end_frame, output_resolution, export_raw, export_npy, export_png)

        batches = np.arange(start_frame, min(end_frame + batch_size, len(self.video)), batch_size)
        left, right = batches[0:-1], batches[1: ]

        for l, r in (pbar := tqdm(zip(left, right))):
            t = slice(l, r)
            run_stats["frame"].extend(range(t.start, t.stop))
            pbar.set_description(f"[VRE] t=[{l}:{r}]")
            for name, representation in self.tsr.items():
                pbar.set_description(f"[VRE] {name}")
                logger.debug2(f"t={t} representation={name}")
                now = datetime.now()
                raw_data = representation(t)
                took = round((datetime.now() - now).total_seconds(), 3)
                run_stats[name].extend([took / (r - l)] * (r - l))

                # for each representation we have 3 cases: raw npy, resized npy and resized png.
                # the raw npy is always computed unless it was previously computed, in which case we load it from
                #  the disk. The assumption is that it is ran with the same cfg on the same video (cli args)
                # raw_path, rsz_path, img_path = (npy_raw_paths[name] / f"{t}.npz", npy_resz_paths[name] / f"{t}.npz",
                #                                 png_resz_paths[name] / f"{t}.png")

                # if raw_path.exists():
                #     raw_data = np.load(raw_path, allow_pickle=True)["arr_0"].item()
                #     run_stats[name].append(np.nan)
                # else:
                #     now = datetime.now()
                #     if not isinstance(representation, Representation) and isinstance(representation, Callable):
                #         logger.info(f"Instantiating representation '{name}'")
                #         representation = representation()
                #         self.tsr[name] = representation
                #     raw_data = representation(t)
                #     run_stats[name].append(round((datetime.now() - now).total_seconds(), 3))
                # if export_raw and not raw_path.exists():
                #     np.savez(raw_path, raw_data)
                # assert isinstance(raw_data, dict) and "data" in raw_data and "extra" in raw_data, raw_data
                # assert isinstance(raw_data["extra"], dict), raw_data

                # for resized npy and resized png we only compute them if these params are provided
                # in the same manner, if they were previosuly computed, we skip them
                # if export_npy and not rsz_path.exists():
                #     rsz_data = representation.resize(raw_data, *output_resolution, only_uint8=False)
                #     np.savez(rsz_path, rsz_data)

                # if export_png and not img_path.exists():
                #     img = representation.make_image(raw_data)
                #     img_resized = cv2.resize(img, output_resolution[::-1], interpolation=cv2.INTER_LINEAR)
                #     cv2.imwrite(str(img_path), img_resized[..., ::-1])

        run_stats = pd.DataFrame(run_stats)
        run_stats.to_csv(output_dir / "run_stats.csv")
        return run_stats

    def __str__(self) -> str:
        return (
            f"VRE ({len(self.tsr)} representations). "
            f"Video: '{self.video.raw_data.path}' (shape: {self.video.shape})"
        )

    def __repr__(self) -> str:
        return str(self)
