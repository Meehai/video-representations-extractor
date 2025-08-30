#!/usr/bin/env python3
"""vre_streaming -- Tool that 'streams' a VRE frame (or batch) by frame to other external tools like ffmpeg or mpl"""
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any
from pathlib import Path
from io import FileIO
import socket
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from vre_video import VREVideo

from vre import VRE, ReprOut
from vre.representations import build_representations_from_cfg, LearnedRepresentationMixin
from vre.logger import vre_logger as logger
from vre.utils import collage_fn, make_batches, image_resize, image_add_title, mean
from vre_repository import get_vre_repository

os.environ["VRE_COLORIZE_SEMSEG_FAST"] = "1"
os.environ["VRE_PBAR"] = "0"

IntOrError = tuple[int | None, Exception | None]

def _get_imgs(res: dict[str, ReprOut]) -> list[np.ndarray]:
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

def build_reader_kwargs(args: Namespace) -> dict[str, Any]:
    """builds the video kwargs that's passed to the VREVideo FrameReader"""
    if args.video_path == "-":
        return {"resolution": args.input_size, "async_worker": not args.disable_async_worker}
    elif args.video_path.startswith("tcp://"):
        return {"resolution": args.input_size, "async_worker": not args.disable_async_worker}
    else:
        assert args.input_size is None, "--input_size cannot be set for video_path that's not stdin"
    return {}

# TODO: simplify and use process one frame.
def process_one_batch(vre: VRE, batch: list[int], output_size: tuple[int, int],
                      disable_title_hud: bool=False, curr_fps: list[float] | None = None,
                      write_buffer: FileIO | None | plt.Figure = None) -> IntOrError:
    curr_fps = curr_fps or []
    write_buffer = write_buffer or sys.stdout.buffer

    now = datetime.now()
    try:
        res = vre[batch]
    except StopIteration as e:
        logger.info(f"StopIteration raised at {batch=}. Exitting.")
        return None, e
    imgs = _get_imgs(res)
    for i, img in enumerate(imgs):
        if not disable_title_hud:
            title = f"Frame: {batch[i]}."
            title = title if curr_fps is None else f"{title} FPS {mean(curr_fps):.2f}"
            img = image_add_title(img, title)
        img = image_resize(img, *output_size)
        if isinstance(write_buffer, plt.Figure):
            plt.imshow(img)
            plt.draw()
            plt.pause(0.00001)
            plt.clf()
        else:
            try:
                write_buffer.write(img.tobytes())
                write_buffer.flush()
            except BrokenPipeError as e:
                return None, e
    return (datetime.now() - now).total_seconds(), None

def get_args() -> Namespace:
    """cli args"""
    parser = ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("config_path", type=Path)
    # generic parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_size", nargs=2, type=int, help="h, w", default=[360, 1280])
    parser.add_argument("--disable_title_hud", action="store_true")
    parser.add_argument("--output_destination", choices=["matplotlib", "socket", "stdout"], default="stdout")
    # parameters for video_path='-' or video_path='tcp://ip:port'
    parser.add_argument("--input_size", nargs=2, type=int)
    parser.add_argument("--disable_async_worker", action="store_true",
                        help="Set this to true for sync streaming like reading from an mp4. Keep it on for real time")
    args = parser.parse_args()
    assert args.batch_size > 0, args.batch_size
    assert all(hw > 0 for hw in args.output_size), args.output_size
    if args.video_path == "-" or args.video_path.startswith("tcp://"):
        assert args.input_size is not None
    return args

def main(args: Namespace):
    """main fn"""

    logger.info(f"Output destination: '{args.output_destination}'")
    if args.video_path.startswith("tcp://"):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        url, port = args.video_path.removeprefix("tcp://").split(":")
        logger.info(f"Listening on {url}:{port}")
        server_sock.bind((url, int(port)))
        server_sock.listen()
        conn, addr = server_sock.accept()
        logger.info(f"Accepted connection from {addr}")
        video_path = conn.makefile("rwb")
    else:
        video_path = args.video_path

    video = VREVideo(video_path, **build_reader_kwargs(args))
    logger.debug(video)
    representations = build_representations_from_cfg(args.config_path, get_vre_repository())

    for r in [_r for _r in representations if isinstance(_r, LearnedRepresentationMixin)]:
        r.vre_setup()
    vre = VRE(video, representations)
    vre.set_compute_params(batch_size=1)
    vre.set_io_parameters(image_format="png", output_size=video.shape[1:3], binary_format="npz", compress=False)

    if args.output_destination == "matplotlib":
        write_buffer = plt.figure(figsize=(12, 6))
        plt.ion()
        plt.show()
    elif args.output_destination == "stdout":
        write_buffer = sys.stdout.buffer
    elif args.output_destination == "socket":
        assert hasattr(video_path, "write") and hasattr(video_path, "flush"), type(video_path)
        write_buffer = video_path

    batches = make_batches(list(range(len(video))), batch_size=1) # note: add here a start_index for testing
    fps = [0]
    while True:
        for bix in batches:
            took_s, err = process_one_batch(vre, bix, args.output_size, args.disable_title_hud, fps, write_buffer)
            if err is not None:
                raise err
            if (diff := (1 / video.fps) - took_s) > 0:
                time.sleep(diff)
            fps = [*fps[-10:], len(bix) / took_s]

if __name__ == "__main__":
    main(get_args())
