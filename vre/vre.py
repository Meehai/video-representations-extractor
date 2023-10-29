"""Video Representations Extractor module"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
from tqdm import trange
import pims
from media_processing_lib.image import image_write, image_resize
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
from functools import partial
from datetime import datetime

from .representations import Representation, build_representation_from_cfg
from .logger import logger
from .utils import topological_sort

RepresentationsBuildDict = Dict[str, DictConfig]  # this is used to build the representations


class VRE:
    """Video Representations Extractor class"""

    def __init__(self, video: pims.Video, representations: dict[str, Representation] = None,
                 representations_dict: RepresentationsBuildDict = None):
        """
        Parameters
        video The video we are performing VRE one
        representations_dict The dict of uninstantiated representations with their constructor args
        """
        # fmt: off
        assert (representations is not None) + (representations_dict is not None) == 1, \
            "Either provide a topo sorted list of representations or a config to build the representations, not both"
        assert isinstance(video, pims.Video), type(video)
        self.video = video

        if representations is not None:
            self._tsr = representations
        else:
            representations_dict = DictConfig(representations_dict)
            assert len(representations_dict) > 0 and isinstance(representations_dict, DictConfig), representations_dict
            self.representations_dict = representations_dict
            self._tsr: Dict[str, Representation] = None

    @property
    def tsr(self) -> Dict[str, Representation]:
        """Call the representations module to build each of them, one by one, after topo sorting for dependencies"""
        if self._tsr is None:
            logger.debug("Doing topological sort...")
            res, dep_graph = {}, {}
            for repr_name, repr_cfg_values in self.representations_dict.items():
                assert isinstance(repr_cfg_values, DictConfig), f"{repr_name} not a dict cfg: {type(repr_cfg_values)}"
                dep_graph[repr_name] = repr_cfg_values["dependencies"]
            topo_sorted = {k: self.representations_dict[k] for k in topological_sort(dep_graph)}
            for name, r in topo_sorted.items():
                obj = build_representation_from_cfg(self.video, r, name, res)
                res[name] = obj
            self._tsr = res
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
            OmegaConf.save(self.representations_dict[name], output_dir / name / "cfg.yaml")
            if export_raw:
                npy_raw_paths[name] = output_dir / name / "npy/raw"
                npy_raw_paths[name].mkdir(exist_ok=True, parents=True)
            if export_npy:
                npy_resz_paths[name] = output_dir / name / f"npy/{out_resolution_str}"
                npy_resz_paths[name].mkdir(exist_ok=True)
            if export_png:
                png_resz_paths[name] = output_dir / name / f"png/{out_resolution_str}"
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
  - Representations ({len(self.representations_dict)}): {", ".join([x for x in self.representations_dict.keys()])}
  - Video shape: {self.video.shape}
  - Output frames ({end_frame - start_frame}): [{start_frame} : {end_frame - 1}]
  - Output resolution: {tuple(output_resolution)}
  - Export raw npy: {export_raw}
  - Export resized npy: {export_npy}
  - Export resized png: {export_png}
"""
        )

    def __call__(self, output_dir: Path, start_frame: int | None = None, end_frame: int | None = None,
                 export_raw: bool | None = False, export_npy: bool | None = False, export_png: bool | None = False,
                 output_resolution: tuple[int, int] | None = None):
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

        for t in (pbar := trange(start_frame, end_frame)):
            run_stats["frame"].append(t)
            for name, representation in self.tsr.items():
                pbar.set_description(f"[VRE] {name}")
                logger.debug2(f"t={t} representation={name}")
                # for each representation we have 3 cases: raw npy, resized npy and resized png.
                # the raw npy is always computed unless it was previously computed, in which case we load it from
                #  the disk. The assumption is that it is ran with the same cfg on the same video (cli args)
                raw_path, rsz_path, img_path = npy_raw_paths[name] / f"{t}.npz", npy_resz_paths[name] / f"{t}.npz", \
                                               png_resz_paths[name] / f"{t}.png"

                if raw_path.exists():
                    raw_data = np.load(raw_path, allow_pickle=True)["arr_0"].item()
                    run_stats[name].append(np.nan)
                else:
                    now = datetime.now()
                    raw_data = representation(t)
                    run_stats[name].append(round((datetime.now() - now).total_seconds(), 3))
                if export_raw and not raw_path.exists():
                    np.savez(raw_path, raw_data)
                assert isinstance(raw_data, dict) and "data" in raw_data and "extra" in raw_data, raw_data
                assert isinstance(raw_data["extra"], dict), raw_data

                # for resized npy and resized png we only compute them if these params are provided
                # in the same manner, if they were previosuly computed, we skip them
                if export_npy and not rsz_path.exists():
                    rsz_data = representation.resize(raw_data, *output_resolution, only_uint8=False)
                    np.savez(rsz_path, rsz_data)

                if export_png and not img_path.exists():
                    img = representation.make_image(raw_data)
                    img_resized = image_resize(img, *output_resolution, only_uint8=False)
                    image_write(img_resized, img_path)

            pd.DataFrame(run_stats).to_csv(output_dir / "run_stats.csv")

    def __str__(self) -> str:
        return (
            f"VRE ({len(self.representations_dict)} representations). "
            f"Video: '{self.video.raw_data.path}' (shape: {self.video.shape})"
        )

    def __repr__(self) -> str:
        return str(self)
