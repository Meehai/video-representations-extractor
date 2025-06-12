#!/usr/bin/env python3
import torch as tr
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from contexttimer import Timer
from tempfile import TemporaryDirectory
from tqdm import tqdm
import os
import time
from datetime import datetime

from vre import VRE, FFmpegVideo, ReprOut, Representation, MemoryData
from vre.logger import vre_logger as logger
from vre.vre_runtime_args import VRERuntimeArgs
from vre.representations import IORepresentationMixin, LearnedRepresentationMixin
from vre.representations.io_representation_mixin import ImageFormat
from vre.utils import get_project_root, collage_fn, random_chars, make_batches, image_resize, image_add_title

from vre_repository.color.rgb import RGB
from vre_repository.semantic_segmentation.safeuav import SafeUAV

os.environ["VRE_DEVICE"] = device = "cuda" if tr.cuda.is_available() else "cpu"
os.environ["VRE_COLORIZE_SEMSEG_FAST"] = "1"

def mean(x):
    return sum(x) / len(x)

def get_imgs(res: dict[str, ReprOut]) -> list[np.ndarray]:
    repr_names = list(res.keys())
    if all(res[r].output_images is None for r in res):
        print("Image format not set, not images were computed. Skipping.")
        return [None] * res[repr_names[0]].key

    frames_res = []
    for i, frame_ix in enumerate(res[repr_names[0]].key):
        imgs = [res[r].output_images[i] if res[r].output_images is not None else None for r in repr_names]
        collage = collage_fn(imgs, titles=repr_names, size_px=70, rows_cols=None) if len(imgs) > 1 else imgs[0]
        frames_res.append(collage)
    return frames_res

def getitem1(vre: VRE, ixs: list[int]) -> dict[str, ReprOut]:
    """wrapper since getitem cannot receive extra args. This is the old 'VREStreaming' code basically."""
    output_dir = Path(TemporaryDirectory(prefix="vre_getitem").name)
    run_id = random_chars(n=10) # we need a temp run id for getitem in "streaming" mode

    res: dict[str, ReprOut] = {}
    for vre_repr in vre.representations:
        assert isinstance(vre_repr, IORepresentationMixin), (vre_repr, type(vre_repr))
        (output_dir / vre_repr.name).mkdir(exist_ok=True, parents=True)
        runtime_args = VRERuntimeArgs(video=vre.video, representations=vre.representations, frames=ixs,
                                      exception_mode="stop_execution", n_threads_data_storer=1)
        _ = vre.do_one_representation(run_id=run_id, representation=vre_repr, output_dir=output_dir,
                                      output_dir_exists_mode="skip_computed", runtime_args=runtime_args)
        res[vre_repr.name] = vre._load_from_disk_if_possible(vre_repr, vre.video,
                                                              ixs=list(ixs), output_dir=output_dir)
        if vre_repr.image_format != ImageFormat.NOT_SET:
            res[vre_repr.name].output_images = vre_repr.make_images(res[vre_repr.name])
    return res

def getitem2(vre: VRE, ixs: list[int]) -> dict[str, ReprOut]:
    """trying to implement vre streaming w/o wrapping the batched functions"""
    assert all(isinstance(vre_repr, IORepresentationMixin) for vre_repr in vre.representations), vre.representations
    res: dict[str, ReprOut] = {}
    for vre_repr in (pbar := tqdm(vre.representations)):
        pbar.set_description(f"[VRE Streaming] {vre_repr.name}")
        if isinstance(vre_repr, LearnedRepresentationMixin) and not vre_repr.setup_called:
            vre_repr.vre_setup()
        dep_names = [r.name for r in vre_repr.dependencies]
        batches = make_batches(ixs, vre_repr.batch_size)
        batch_res: list[ReprOut] = []
        for b_ixs in batches:
            repr_out = vre_repr.compute(video=vre.video, ixs=b_ixs, dep_data=[res[dep_name] for dep_name in dep_names])
            batch_res.append(vre_repr.resize(repr_out, vre.video.shape[1:3]))
        combined = ReprOut(frames=np.concatenate([br.frames for br in batch_res]),
                           key=sum([br.key for br in batch_res], []),
                           output=MemoryData(np.concatenate([br.output for br in batch_res])))
        combined.output_images = vre_repr.make_images(combined)
        res[vre_repr.name] = combined
    return res

def compare_variants(vre: VRE, frames: list[int]):
    with Timer(prefix="getitem1"):
        res_getitem1 = getitem1(vre, frames)
    with Timer(prefix="getitem2"):
        res_getitem2 = getitem2(vre, frames)
    with Timer(prefix="__getitem__"):
        res_getitem = vre[frames]

    for k in res_getitem.keys():
        for other in [res_getitem1, res_getitem2]:
            assert res_getitem.keys() == other.keys(), (res_getitem.keys(), other.keys())
            assert ((res_getitem[k].output - other[k].output).__abs__() < 0.05).all(), k
            assert ((res_getitem[k].output_images - other[k].output_images).__abs__().mean() < 1e-3), k

def main():
    video = FFmpegVideo(get_project_root() / "resources/test_video.mp4", cache_len=100)
    logger.debug(video)

    representations: list[Representation] = [
        RGB(name="rgb", dependencies=[]),
        safeuav := SafeUAV(name="safeuav", dependencies=[], disk_data_argmax=False, variant="model_150k"),
    ]
    safeuav.device = device
    safeuav.vre_setup()
    vre = VRE(video, representations)
    vre.set_compute_params(batch_size=1)
    vre.set_io_parameters(image_format="png", output_size=video.shape[1:3], binary_format="npz", compress=False)
    vre.to_graphviz().render("graph", format="png", cleanup=True)
    # frames = [100, 1000, 3000]

    # compare_variants(vre, frames)

    if os.getenv("MPL", "0") == "1":
        plt.figure()
        plt.ion()
        plt.show()
    # batches = make_batches(list(range(1000, len(video))), batch_size=1)
    fps = [0]
    for ix in range(1000, len(video)):
        now = datetime.now()
        res = vre[ix]
        img = get_imgs(res)[0]
        # img = image_resize(img, height=360, width=1280)
        img = image_resize(image_add_title(img, f"Frame: {ix}. FPS: {mean(fps):.2f}"), height=360, width=1280)
        if os.getenv("MPL", "0") == "1":
            plt.imshow(img)
            plt.draw()
            plt.pause(0.00001)
            plt.clf()
        else:
            # sys.stderr.write(f"{img.shape}, {img.dtype}\n")
            sys.stdout.buffer.write(img.tobytes())
            sys.stdout.flush()

        if (diff := (1 / video.fps) - (datetime.now() - now).total_seconds()) > 0:
            time.sleep(diff)
        fps.append(1 / (datetime.now() - now).total_seconds())

if __name__ == "__main__":
    main()
