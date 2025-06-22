#!/usr/bin/env python3
"""vre_streaming -- Tool that 'streams' a VRE frame (or batch) by frame to other external tools like ffmpeg or mpl"""
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from vre_video import VREVideo

from vre import VRE, ReprOut
from vre.representations import build_representations_from_cfg, LearnedRepresentationMixin
from vre.logger import vre_logger as logger
from vre.utils import collage_fn, make_batches, image_resize, image_add_title
from vre_repository import get_vre_repository

os.environ["VRE_COLORIZE_SEMSEG_FAST"] = "1"
os.environ["VRE_PBAR"] = "0"

def mean(x):
    return sum(x) / len(x)

def get_imgs(res: dict[str, ReprOut]) -> list[np.ndarray]:
    repr_names = list(res.keys())
    if all(res[r].output_images is None for r in res):
        print("Image format not set, not images were computed. Skipping.")
        return [None] * res[repr_names[0]].key

    frames_res = []
    for i in range(len(res[repr_names[0]].key)):
        imgs = [res[r].output_images[i] if res[r].output_images is not None else None for r in repr_names]
        collage = collage_fn(imgs, titles=repr_names, size_px=70, rows_cols=None) if len(imgs) > 1 else imgs[0]
        frames_res.append(collage)
    return frames_res

def get_args() -> Namespace:
    """cli args"""
    parser = ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", nargs=2, type=int, help="h, w")
    parser.add_argument("--output_size", nargs=2, type=int, help="h, w", default=[360, 1280])
    parser.add_argument("--disable_title_hud", action="store_true")
    args = parser.parse_args()
    assert args.batch_size > 0, args.batch_size
    assert all(hw > 0 for hw in args.output_size), args.output_size
    return args

def main(args: Namespace):
    """main fn"""
    video_kwargs = {"resolution": args.input_size} if args.video_path == "-" else {}
    video = VREVideo(args.video_path, **video_kwargs)
    logger.debug(video)
    representations = build_representations_from_cfg(args.config_path, get_vre_repository())

    for r in [_r for _r in representations if isinstance(_r, LearnedRepresentationMixin)]:
        r.vre_setup()
    vre = VRE(video, representations)
    vre.set_compute_params(batch_size=1)
    vre.set_io_parameters(image_format="png", output_size=video.shape[1:3], binary_format="npz", compress=False)
    vre.to_graphviz().render("graph", format="png", cleanup=True)

    if os.getenv("MPL", "0") == "1":
        plt.figure(figsize=(12, 6))
        plt.ion()
        plt.show()
    batches = make_batches(list(range(len(video))), batch_size=1)
    fps = [0]
    while True:
        for bix in batches:
            now = datetime.now()
            try:
                res = vre[bix]
            except StopIteration:
                logger.info(f"StopIteration raised at {bix=}. Exitting.")
                exit(0)
            imgs = get_imgs(res)
            for i in range(len(bix)):
                img = imgs[i]
                if not args.disable_title_hud:
                    img = image_add_title(imgs[i], f"Frame: {bix[i]}. FPS: {mean(fps):.2f}")
                img = image_resize(img, *args.output_size)
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
            fps = [*fps[-10:], len(bix) / (datetime.now() - now).total_seconds()]

if __name__ == "__main__":
    main(get_args())
