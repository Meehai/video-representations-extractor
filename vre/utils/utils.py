"""utils for vre"""
from typing import Any
from pathlib import Path
from datetime import datetime
import os
import gdown
import numpy as np
from skimage.transform import resize
from skimage.io import imsave, imread
from skimage.color import hsv2rgb

from ..logger import logger
from .fake_video import VREVideo

RepresentationOutput = np.ndarray | tuple[np.ndarray, list[dict]]

def get_project_root() -> Path:
    """gets the root of this project"""
    return Path(__file__).parents[2].absolute()

def get_weights_dir() -> Path:
    """gets the weights dir of this project"""
    return Path(os.getenv("VRE_WEIGHTS_DIR", get_project_root() / "weights"))

def parsed_str_type(item: Any) -> str:
    """Given an object with a type of the format: <class 'A.B.C.D'>, parse it and return 'A.B.C.D'"""
    return str(type(item)).rsplit(".", maxsplit=1)[-1][0:-2]

def image_resize(data: np.ndarray, height: int, width: int, interpolation: str = "bilinear", **kwargs) -> np.ndarray:
    """Skimage image resizer"""
    assert interpolation in ("nearest", "bilinear", "bicubic", "biquadratic", "biquartic", "biquintic")
    assert isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0, (height, width)
    # As per: https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_warps.py#L820
    order = {"nearest": 0, "bilinear": 1, "biquadratic": 2, "bicubic": 3, "biquartic": 4, "biquintic": 5}[interpolation]
    img_resized = resize(data, output_shape=(height, width), order=order, preserve_range=True, **kwargs)
    img_resized = img_resized.astype(data.dtype)
    return img_resized

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

def took(prev: datetime.date, l: int, r: int) -> list[float]:
    """how much it took between [prev:now()] gien a [l:r] batch"""
    return [(datetime.now() - prev).total_seconds() / (r - l)] * (r - l)

def make_batches(video: VREVideo, start_frame: int, end_frame: int, batch_size: int) -> np.ndarray:
    """return 1D array [start_frame, start_frame+bs, start_frame+2*bs... end_frame]"""
    if batch_size > end_frame - start_frame:
        logger.warning(f"batch size {batch_size} is larger than #frames to process [{start_frame}:{end_frame}].")
        batch_size = end_frame - start_frame
    last_one = min(end_frame, len(video))
    batches = np.arange(start_frame, last_one, batch_size)
    batches = np.array([*batches, last_one], dtype=np.int64) if batches[-1] != last_one else batches
    return batches

def all_batch_exists(npy_paths: list[Path], png_paths: list[Path], l: int, r: int,
                     export_npy: bool, export_png: bool) -> bool:
    """checks whether all batches exist or not"""
    for ix in range(l, r):
        if export_npy and not npy_paths[ix].exists():
            return False
        if export_png and not png_paths[ix].exists():
            return False
    logger.debug(f"Batch [{l}:{r}] skipped.")
    return True

def is_dir_empty(dir_path: Path, pattern: str = "*") -> bool:
    """returns true if directory is not empty, false if it is empty"""
    return len(list(dir_path.glob(pattern))) == 0
