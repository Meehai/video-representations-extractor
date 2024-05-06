"""Video Representations Extractor module"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from functools import reduce
import traceback
from tqdm import tqdm
import numpy as np
import torch as tr
import pandas as pd

from .representation import Representation, RepresentationOutput
from .utils import image_write, VREVideo, took, make_batches, all_batch_exists
from .vre_runtime_args import VRERuntimeArgs
from .logger import logger

def _open_write_err(path: str, msg: str):
    open(path, "a").write(msg)
    logger.debug(f"Error: {msg}")

def _vre_make(vre: VideoRepresentationsExtractor, _repr: Representation, ix: slice, make_images: bool) \
        -> (RepresentationOutput, np.ndarray | None):
    """
    Method used to integrate with VRE. Gets the entire data (video) and a slice of it (ix) and returns the
    representation for that slice. Additionally, if makes_images is set to True, it also returns the image
    representations of this slice.
    """
    if tr.cuda.is_available():
        tr.cuda.empty_cache()
    frames = np.array(vre.video[ix])
    dep_data = _repr.vre_dep_data(vre.video, ix)
    res = _repr.make(frames, **dep_data)
    repr_data, extra = res if isinstance(res, tuple) else (res, {})
    imgs = _repr.make_images(frames, res) if make_images else None
    return (repr_data, extra), imgs

class VideoRepresentationsExtractor:
    """Video Representations Extractor class"""

    def __init__(self, video: VREVideo, representations: dict[str, Representation], output_dir: Path):
        """
        Parameters:
        - video The video we are performing VRE one
        - representations The dict of instantiated and topo sorted representations (or callable to instantiate them)
        - output_dir The directory where the data is stored
        """
        assert len(representations) > 0, "At least one representation must be provided"
        assert all(lambda x: isinstance(x, Representation) for x in representations.values()), representations
        self.video = video
        self.representations: dict[str, Representation] = representations
        self.output_dir = output_dir

    def _store_data(self, name: str, raw_data: np.ndarray, extra: dict, imgs: np.ndarray | None,
                    l: int, r: int, runtime_args: VRERuntimeArgs):
        """store the data in the right format"""
        h, w = self.video.frame_shape[0:2]
        if runtime_args.export_png:
            assert imgs is not None
            assert imgs.shape == (r - l, h, w, 3), (imgs.shape, (r - l, h, w, 3))
            assert imgs.dtype == np.uint8, imgs.dtype

        for i, t in enumerate(range(l, r)):
            if runtime_args.export_npy:
                if not runtime_args.npy_paths[name][t].exists():
                    np.savez(runtime_args.npy_paths[name][t], raw_data[i])
                    if len(extra) > 0:
                        np.savez(runtime_args.npy_paths[name][t].parent / f"{t}_extra.npz", extra[i])
            if runtime_args.export_png:
                image_write(imgs[i], runtime_args.png_paths[name][t])

    def _do_one_representation(self, representation: Representation, runtime_args: VRERuntimeArgs):
        """main loop of each representation."""
        name = representation.name
        batch_size = runtime_args.batch_sizes[name]
        npy_paths, png_paths = runtime_args.npy_paths[name], runtime_args.png_paths[name]

        # call vre_setup here so expensive representations get lazy deep instantiated (i.e. models loading)
        try:
            representation.vre_setup(video=self.video, **runtime_args.reprs_setup[name])
        except Exception:
            _open_write_err("exception.txt", f"\n[{name} {datetime.now()} {batch_size=} {traceback.format_exc()}\n")
            del representation
            return {name: [1 << 31] * (runtime_args.end_frame - runtime_args.start_frame)}

        batches = make_batches(self.video, runtime_args.start_frame, runtime_args.end_frame, batch_size)
        left, right = batches[0:-1], batches[1:]
        repr_stats = []
        pbar = tqdm(total=runtime_args.end_frame - runtime_args.start_frame, desc=f"[VRE] {name} bs={batch_size}")
        for l, r in zip(left, right): # main VRE loop
            if all_batch_exists(npy_paths, png_paths, l, r, runtime_args.export_npy, runtime_args.export_png):
                pbar.update(r - l)
                repr_stats.extend(took(datetime.now(), l, r))
                continue

            now = datetime.now()
            try:
                (raw_data, extra), imgs = _vre_make(self, representation, slice(l, r), runtime_args.export_png) # noqa
                self._store_data(name, raw_data, extra, imgs, l, r, runtime_args)
            except Exception:
                _open_write_err("exception.txt", f"\n[{name} {now} {batch_size=} {l=} {r=}] {traceback.format_exc()}\n")
                repr_stats.extend([1 << 31] * (runtime_args.end_frame - l))
                del representation # noqa
                break
            # update the statistics and the progress bar
            repr_stats.extend(took(now, l, r))
            pbar.update(r - l)
        return {name: repr_stats}

    def run(self, start_frame: int | None = None, end_frame: int | None = None, batch_size: int = 1,
            export_npy: bool = True, export_png: bool = True, reprs_setup: dict | None = None,
            output_dir_exist_mode: str = "raise", exception_mode: str = "stop_execution"):
        """
        The main loop of the VRE. This will run all the representations on the video and store results in the output_dir
        Parameters:
        - start_frame The first frame to process (inclusive). If not provided, defaults to 0.
        - end_frame The last frame to process (inclusive). If not provided, defaults to len(video).
        - batch_size The batch size to use when processing the video. If not provided, defaults to 1.
        - export_npy Whether to export the npy files
        - export_png Whether to export the png files
        - reprs_setup A dict of {representation_name: {representation_setup}}. This is used to pass
            representation specific inference parameters to the VRE, like setting device before running or loading
            some non-standard weights.
        - output_dir_exist_mode What to do if the output dir already exists. Can be one of:
          - 'overwrite' Overwrite the output dir if it already exists
          - 'skip_computed' Skip the computed frames and continue from the last computed frame
          - 'raise' Raise an error if the output dir already exists
          Defaults to 'raise'.
        - exception_mode What to do when encountering an exception. It always writes the exception to 'exception.txt'.
          - 'skip_representation' Will stop the run of the current representation and start the next one
          - 'stop_execution' Will stop the execution of VRE
          Defaults to 'stop_execution'
        Returns:
        - A dataframe with the run statistics for each representation
        """
        runtime_args = VRERuntimeArgs(self, start_frame, end_frame, batch_size, export_npy, export_png, reprs_setup,
                                      output_dir_exist_mode, exception_mode)
        run_stats = []
        for name, vre_repr in self.representations.items():
            repr_res = self._do_one_representation(vre_repr, runtime_args)
            if repr_res[name][-1] == 1 << 31 and runtime_args.exception_mode == "stop_execution":
                raise RuntimeError(f"Representation '{name}' threw. Check 'exceptions.txt' for information")
            run_stats.append(repr_res)
            del vre_repr
        df_run_stats = pd.DataFrame(reduce(lambda a, b: {**a, **b}, run_stats),
                                    index=range(runtime_args.start_frame, runtime_args.end_frame))
        return df_run_stats

    # pylint: disable=too-many-branches, too-many-nested-blocks
    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.run(*args, **kwargs)

    def __str__(self) -> str:
        return (
            f"VRE ({len(self.representations)} representations). "
            f"Video: '{self.video.file}' (shape: {self.video.shape})"
        )

    def __repr__(self) -> str:
        return str(self)
