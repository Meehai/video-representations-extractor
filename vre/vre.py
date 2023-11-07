"""Video Representations Extractor module"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os
from tqdm import tqdm
from omegaconf import DictConfig
import pims
import numpy as np
import pandas as pd

from .representation import Representation
from .logger import logger
from .utils import image_resize, image_write

RunPaths = tuple[dict[str, list[Path]], dict[str, list[Path]], dict[str, list[Path]]]

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
        self.representations = representations

    def _make_run_paths(self, output_dir: Path, output_resolution: tuple[int, int],
                        export_raw: bool, export_npy: bool, export_png: bool) -> RunPaths:
        """
        create the output dirs structure. We may have 2 types of outputs: npy and png for each, we may have
        different resolutions, so we store them under npy/HxW or png/HxW for the npy we always store the raw output
        under 'raw' from which we can derive all the others even at a later time
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_resolution_str = f"{output_resolution[0]}x{output_resolution[1]}"

        npy_raw_paths, npy_resz_paths, png_resz_paths = {}, {}, {}
        for name in self.representations.keys():
            (output_dir / name).mkdir(exist_ok=True, parents=True)
            npy_raw_base_dir = output_dir / name / "npy/raw"
            npy_resz_base_dir = output_dir / name / f"npy/{out_resolution_str}"
            png_resz_base_dir = output_dir / name / f"png/{out_resolution_str}"

            npy_raw_paths[name] = [npy_raw_base_dir / f"{t}.npz" for t in range(len(self.video))]
            npy_resz_paths[name] = [npy_resz_base_dir / f"{t}.npz" for t in range(len(self.video))]
            png_resz_paths[name] = [png_resz_base_dir / f"{t}.png" for t in range(len(self.video))]

            if export_raw:
                npy_raw_base_dir.mkdir(exist_ok=True, parents=True)
            if export_npy:
                npy_resz_base_dir.mkdir(exist_ok=True)
            if export_png:
                png_resz_base_dir.mkdir(exist_ok=True, parents=True)
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
  - Representations ({len(self.representations)}): {", ".join([x for x in self.representations.keys()])}
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
            logger.warning("start frame not set, default to 0")
        assert isinstance(start_frame, int) and start_frame <= end_frame, (start_frame, end_frame)
        Path(f"{os.environ['VRE_WEIGHTS_DIR']}").mkdir(exist_ok=True, parents=True)
        # run_stats will hold a dict: {repr_name: [time_taken, ...]} for all representations, for debugging/logging
        run_stats: dict[str, list] = {repr_name: [] for repr_name in ["frame", *self.representations.keys()]}

        npy_raw_paths, npy_resz_paths, png_resz_paths = self._make_run_paths(output_dir, output_resolution,
                                                                             export_raw, export_npy, export_png)
        self._print(output_dir, start_frame, end_frame, output_resolution, export_raw, export_npy, export_png)

        batches = np.arange(start_frame, min(end_frame + batch_size, len(self.video)), batch_size)
        left, right = batches[0:-1], batches[1: ]

        pbar = tqdm(total=end_frame - start_frame)
        for l, r in zip(left, right):
            batch_t = slice(l, r)
            run_stats["frame"].extend(range(batch_t.start, batch_t.stop))
            pbar.update(r - l)
            representation: Representation
            for name, representation in self.representations.items():
                pbar.set_description(f"[VRE] {name}")
                now = datetime.now()
                # TODO: if all paths exist, skip and read from disk
                raw_data, extra = representation[batch_t]
                took = round((datetime.now() - now).total_seconds(), 3)
                run_stats[name].extend([took / (r - l)] * (r - l))

                if export_raw:
                    for i, t in enumerate(range(l, r)):
                        if not npy_raw_paths[name][t].exists():
                            np.savez(npy_raw_paths[name][t], raw_data[i])
                            if len(extra) > 0:
                                np.savez(npy_raw_paths[name][t].parent / f"{t}_extra.npz", extra[i])

                if export_png:
                    imgs = representation.make_images(raw_data, extra)
                    for i, t in enumerate(range(l, r)):
                        img_resized = image_resize(imgs[i], height=output_resolution[0], width=output_resolution[1])
                        image_write(img_resized, png_resz_paths[name][t])

                if export_npy:
                    rsz_data = representation.resize(raw_data, height=output_resolution[0], width=output_resolution[1])
                    for i, t in enumerate(range(l, r)):
                        if not npy_resz_paths[name][t].exists():
                            np.savez(npy_resz_paths[name][t], rsz_data[i])
        pbar.close()

        df_run_stats = pd.DataFrame(run_stats)
        df_run_stats.to_csv(output_dir / "run_stats.csv")
        return run_stats

    def __str__(self) -> str:
        return (
            f"VRE ({len(self.representations)} representations). "
            f"Video: '{self.video.raw_data.path}' (shape: {self.video.shape})"
        )

    def __repr__(self) -> str:
        return str(self)
