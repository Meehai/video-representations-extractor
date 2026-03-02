"""image.py -- Basic wrapper over image_utils. Mostly a module so we don't do cyclic imports."""
from pathlib import Path
import numpy as np
from PIL import Image
from image_utils import image_resize
from .repr_memory_layout import MemoryData

def image_resize_batch(x_batch: np.ndarray | list[np.ndarray], *args, **kwargs) -> np.ndarray:
    """resizes a bath of images to the given height and width"""
    fn = MemoryData if isinstance(x_batch[0], MemoryData) else np.asarray
    return fn([image_resize(x, *args, **kwargs) for x in x_batch])

def image_read(path: Path) -> np.ndarray:
    """wrapper over pil for image read"""
    return np.array(Image.open(path), dtype=np.uint8)[..., 0:3]

def image_write(image: np.ndarray, path: Path):
    """wrapper over pil for image write"""
    Image.fromarray(image, "RGB").save(path)
