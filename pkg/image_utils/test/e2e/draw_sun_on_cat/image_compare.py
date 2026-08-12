#!/usr/bin/env python3
"""returns the score (0-1) of how close img1 is to img2 ('close' in pixels, not perceptual similarity)"""
from argparse import ArgumentParser, Namespace
from PIL import Image
import numpy as np
from loggez import loggez_logger as logger

def get_args() -> Namespace:
    """Cli args"""
    parser = ArgumentParser()
    parser.add_argument("img1")
    parser.add_argument("img2")
    args = parser.parse_args()
    return args

def main(args: Namespace):
    img1 = np.array(Image.open(args.img1)).astype(np.int64)
    img2 = np.array(Image.open(args.img2)).astype(np.int64)
    logger.info(f"Read image 1: '{args.img1}' {img1.shape} {img1.dtype}")
    logger.info(f"Read image 2: '{args.img2}' {img2.shape} {img2.dtype}")
    diff = (img1 - img2).__abs__().mean() / 255
    res = 1 - diff # keep full rounding
    print(res, flush=True, end="")

if __name__ == "__main__":
    main(get_args())
