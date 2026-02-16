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
from vre_video.readers import FdFrameReader

from vre import VRE, ReprOut
from vre.representations import build_representations_from_cfg, LearnedRepresentationMixin
from vre.logger import vre_logger as logger
from vre.utils import collage_fn, image_resize, image_add_title
from vre_repository import get_vre_repository

os.environ["VRE_COLORIZE_SEMSEG_FAST"] = "1"
os.environ["VRE_PBAR"] = "0"

def _get_img(res: dict[str, ReprOut], disable_hud: bool, title: str, output_size: tuple[int, int]) -> np.ndarray | None:
    repr_names = list(res.keys())
    if all(res[r].output_images is None for r in res):
        print("Image format not set, not images were computed. Skipping.")
        return None
    assert all(len(res[r].key) == 1 for r in repr_names), f"computed for batch >1 which is not allowed: {res}"

    imgs = [res[r].output_images[0] if res[r].output_images is not None else None for r in repr_names]
    if len(imgs) > 1:
        res = collage_fn(imgs, titles=None if disable_hud else repr_names)
    else:
        res = imgs[0] if disable_hud else image_add_title(imgs[0], repr_names[0])
    res = res if disable_hud else image_add_title(res, title)
    res = image_resize(res, *output_size)
    return res

def _build_reader_kwargs(args: Namespace) -> dict[str, Any]:
    """builds the video kwargs that's passed to the VREVideo FrameReader"""
    if args.video_path == "-":
        return {"resolution": args.input_size, "async_worker": not args.disable_async_worker, "fps": args.fps}
    elif args.video_path.startswith("tcp://"):
        return {"resolution": args.input_size, "async_worker": not args.disable_async_worker, "fps": args.fps}
    else:
        assert args.input_size is None, "--input_size cannot be set for video_path that's not stdin"
    return {}

def _write_to_output(write_buffer: plt.Figure | FileIO, img: np.ndarray):
    """writes the final image to the destination buffer or matplotlib"""
    if isinstance(write_buffer, plt.Figure):
        plt.imshow(img)
        plt.draw()
        plt.pause(0.00001)
        plt.clf()
    else:
        write_buffer.write(img.tobytes())
        write_buffer.flush()

def get_args() -> Namespace:
    """cli args"""
    parser = ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("config_path", type=Path)
    # generic parameters
    parser.add_argument("--output_size", nargs=2, type=int, help="h, w", default=[360, 1280])
    parser.add_argument("--disable_hud", action="store_true")
    parser.add_argument("--output_destination", choices=["matplotlib", "socket", "stdout"], default="stdout")
    # parameters for video_path='-' or video_path='tcp://ip:port'
    parser.add_argument("--input_size", nargs=2, type=int)
    parser.add_argument("--disable_async_worker", action="store_true",
                        help="Set this to true for sync streaming like reading from an mp4. Keep it on for real time")
    parser.add_argument("--fps", type=float, help="The FPS of the incoming source. If unknown, set to 30.", default=30)
    args = parser.parse_args()
    assert all(hw > 0 for hw in args.output_size), args.output_size
    if args.video_path == "-" or args.video_path.startswith("tcp://"):
        assert args.input_size is not None
    else:
        assert args.disable_async_worker is False, "should be set only for stdin read"
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

    video = VREVideo(video_path, **_build_reader_kwargs(args))
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
    elif args.output_destination == "socket":
        assert hasattr(video_path, "write") and hasattr(video_path, "flush"), type(video_path)
        write_buffer = video_path
    else:
        write_buffer = sys.stdout.buffer

    ixs = list(range(len(video)))
    while True:
        for ix in ixs:
            now = datetime.now()
            try:
                res = vre[ix]
                if isinstance(video.reader, FdFrameReader): # for streaming, we use the internal counter for frame ix
                    ix = video.reader.current_frame_ix
            except StopIteration as e:
                logger.info(f"StopIteration raised at {ix=}. Exitting.")
                raise e
            img = _get_img(res, args.disable_hud, title=f"Frame: {ix}.", output_size=args.output_size)
            _write_to_output(write_buffer, img)

            if (diff := (1 / video.fps) - (datetime.now() - now).total_seconds()) > 0:
                time.sleep(diff)

if __name__ == "__main__":
    main(get_args())
