"""utils for vre"""
from typing import Any
from pathlib import Path
import os
import gdown
import numpy as np
from skimage.transform import resize
from skimage.io import imsave, imread
from skimage.color import hsv2rgb

from ..logger import logger

def get_project_root() -> Path:
    """gets the root of this project"""
    return Path(__file__).parents[2].absolute()

def get_weights_dir() -> Path:
    """gets the weights dir of this project"""
    return Path(os.getenv("VRE_WEIGHTS_DIR", get_project_root() / "weights"))

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

def image_read(path: str) -> np.ndarray:
    """PIL image reader"""
    image = np.array(imread(path), dtype=np.uint8)[..., 0:3]
    return image

def gdown_mkdir(uri: str, local_path: Path):
    """calls gdown but also makes the directory of the parent path just to be sure it exists"""
    logger.debug(f"Downloading '{uri}' to '{local_path}'")
    local_path.parent.mkdir(exist_ok=True, parents=True)
    gdown.download(uri, f"{local_path}")


def to_categorical(data: np.ndarray, num_classes: int = None) -> np.ndarray:
    """converts the data to categorical. If num classes is not provided, it is infered from the data"""
    data = np.array(data)
    assert data.dtype in (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)
    if num_classes is None:
        num_classes = data.max()
    y = np.eye(num_classes)[data.reshape(-1)].astype(np.uint8)
    # Some bugs for (1, 1) shapes return (1, ) instead of (1, NC)
    MB = data.shape[0]
    y = np.squeeze(y)
    if MB == 1:
        y = np.expand_dims(y, axis=0)
    y = y.reshape(*data.shape, num_classes)
    return y

def generate_diverse_colors(n: int, saturation: float, value: float) -> list[tuple[int, int, int]]:
    """generates a list of n diverse colors using the hue from the HSV transform"""
    assert 0 <= saturation <= 1, saturation
    assert 0 <= value <= 1, value
    colors = []
    for i in range(n):
        hue = i / n  # Vary the hue component
        rgb = hsv2rgb([hue, saturation, value])
        # Convert to 8-bit RGB values (0-255)
        rgb = tuple(int(255 * x) for x in rgb)
        colors.append(rgb)
    return colors
