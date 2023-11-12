"""utils for vre"""
from typing import Any
from pathlib import Path
import gdown
import numpy as np
from skimage.transform import resize
from skimage.io import imsave

from ..logger import logger

def get_project_root() -> Path:
    """gets the root of this project"""
    return Path(__file__).parents[2]

def parsed_str_type(item: Any) -> str:
    """Given an object with a type of the format: <class 'A.B.C.D'>, parse it and return 'A.B.C.D'"""
    return str(type(item)).rsplit(".", maxsplit=1)[-1][0:-2]

def image_resize(x: np.ndarray, height: int, width: int, **kwargs) -> np.ndarray:
    """resizes an image to the given height and width"""
    dtype_orig = x.dtype
    x = x.astype(np.float32) / 255 if dtype_orig == np.uint8 else x
    y = resize(x, output_shape=(height, width), **kwargs)
    y = (y * 255).astype(dtype_orig) if dtype_orig == np.uint8 else y
    return y

def image_resize_batch(x_batch: np.ndarray, height: int, width: int, **kwargs) -> np.ndarray:
    """resizes a bath of images to the given height and width"""
    return np.array([image_resize(x, height, width, **kwargs) for x in x_batch])

def image_write(x: np.ndarray, path: Path):
    """writes an image to a bytes string"""
    assert x.dtype == np.uint8, x.dtype
    imsave(path, x, check_contrast=False)
    logger.debug2(f"Saved image to '{path}'")

def gdown_mkdir(uri: str, local_path: Path):
    """calls gdown but also makes the directory of the parent path just to be sure it exists"""
    logger.debug(f"Downloading '{uri}' to '{local_path}'")
    local_path.parent.mkdir(exist_ok=True, parents=True)
    gdown.download(uri, f"{local_path}")
