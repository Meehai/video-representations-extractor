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

from .representation import Representation
from .utils import VREVideo, took, make_batches, all_batch_exists, RepresentationOutput
from .vre_runtime_args import VRERuntimeArgs
from .data_storer import DataStorer
from .logger import logger

def _open_write_err(path: str, msg: str):
    open(path, "a").write(msg)
    logger.debug(f"Error: {msg}")

class VideoRepresentationsExtractor:
    """Video Representations Extractor class"""

    def __init__(self, video: VREVideo, representations: dict[str, Representation]):
        """
        Parameters:
        - video The video we are performing VRE one
        - representations The dict of instantiated and topo sorted representations (or callable to instantiate them)
        """
        assert len(representations) > 0, "At least one representation must be provided"
        assert all(lambda x: isinstance(x, Representation) for x in representations.values()), representations
        self.video = video
        self.representations: dict[str, Representation] = representations
        self._data_storer: DataStorer | None = None

    def _make_one_frame(self, _repr: Representation, ix: slice, runtime_args: VRERuntimeArgs) \
            -> (RepresentationOutput, np.ndarray | None):
        """
        Method used to integrate with VRE. Gets the entire data (video) and a slice of it (ix) and returns the
        representation for that slice. Additionally, if makes_images is set to True, it also returns the image
        representations of this slice.
        """
        if tr.cuda.is_available():
            tr.cuda.empty_cache()
        frames = np.array(self.video[ix])
        dep_data = _repr.vre_dep_data(self.video, ix)
        res_native = _repr.make(frames, **dep_data)
        if (o_s := runtime_args.output_sizes[_repr.name]) == "native":
            res = res_native
        elif o_s == "video_shape":
            res = _repr.resize(res_native, self.video.frame_shape[0:2])
        else:
            res = _repr.resize(res_native, o_s)
        repr_data, extra = res if isinstance(res, tuple) else (res, {})
        imgs = _repr.make_images(frames, res) if runtime_args.export_png else None
        return (repr_data, extra), imgs

    def _do_one_representation(self, representation: Representation, runtime_args: VRERuntimeArgs):
        """main loop of each representation."""
        name = representation.name
        batch_size = runtime_args.batch_sizes[name]
        npy_paths, png_paths = runtime_args.npy_paths[name], runtime_args.png_paths[name]

        # call vre_setup here so expensive representations get lazy deep instantiated (i.e. models loading)
        try:
            representation.vre_setup(video=self.video, **representation.vre_parameters) # these come from the yaml file
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
                y_repr, imgs = self._make_one_frame(representation, slice(l, r), runtime_args) # noqa
                self._data_storer(name, y_repr, imgs, l, r, runtime_args, self.video.frame_shape[0:2])
            except Exception:
                _open_write_err("exception.txt", f"\n[{name} {now} {batch_size=} {l=} {r=}] {traceback.format_exc()}\n")
                repr_stats.extend([1 << 31] * (runtime_args.end_frame - l))
                del representation # noqa
                break
            # update the statistics and the progress bar
            repr_stats.extend(took(now, l, r))
            pbar.update(r - l)
        return {name: repr_stats}

    def run(self, output_dir: Path, start_frame: int | None = None, end_frame: int | None = None, batch_size: int = 1,
            export_npy: bool = True, export_png: bool = True, output_dir_exist_mode: str = "raise",
            exception_mode: str = "stop_execution", output_size: str | tuple = "video_shape",
            n_threads_data_storer: int = 0) -> pd.DataFrame:
        """
        The main loop of the VRE. This will run all the representations on the video and store results in the output_dir
        See VRERuntimeArgs for parameters definition.
        Returns:
        - A dataframe with the run statistics for each representation
        """
        runtime_args = VRERuntimeArgs(self, output_dir, start_frame, end_frame, batch_size, export_npy, export_png,
                                      output_dir_exist_mode, exception_mode, output_size, n_threads_data_storer)
        self._data_storer = DataStorer(n_threads_data_storer)
        run_stats = []
        for name, vre_repr in self.representations.items():
            repr_res = self._do_one_representation(vre_repr, runtime_args)
            if repr_res[name][-1] == 1 << 31 and runtime_args.exception_mode == "stop_execution":
                raise RuntimeError(f"Representation '{name}' threw. Check 'exceptions.txt' for information")
            run_stats.append(repr_res)
            del vre_repr
        df_run_stats = pd.DataFrame(reduce(lambda a, b: {**a, **b}, run_stats),
                                    index=range(runtime_args.start_frame, runtime_args.end_frame))
        self._data_storer.join_with_timeout(timeout=30)
        return df_run_stats

    # pylint: disable=too-many-branches, too-many-nested-blocks
    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.run(*args, **kwargs)

    def __str__(self) -> str:
        return f"VRE ({len(self.representations)} representations). Video: '{self.video.file}' ({self.video.shape})"

    def __repr__(self) -> str:
        return str(self)
