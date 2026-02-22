"""resize.py -- image resize tools"""
import numpy as np
from .cv2_utils import cv2_image_resize
from .pil_utils import pil_image_resize
from .repr_memory_layout import MemoryData

def image_resize(data: np.ndarray, height: int | None, width: int | None, interpolation: str = "bilinear",
                 library: str = "PIL", **kwargs) -> np.ndarray:
    """image resize. Allows 2 libraries: PIL and cv2 (to alleviate potential pre-trained issues)"""
    assert ((width is None) or width == -1) + ((height is None) or height == -1) <= 1, "At least one must be set"
    _scale = lambda a, b, c: int(b / a * c) # pylint: disable=unnecessary-lambda-assignment
    width = _scale(data.shape[0], height, data.shape[1]) if (width is None or width == -1) else width
    height = _scale(data.shape[1], width, data.shape[0]) if (height is None or height == -1) else height
    assert isinstance(height, int) and isinstance(width, int), (type(height), type(width))
    return {"cv2": cv2_image_resize, "PIL": pil_image_resize}[library](data, height, width, interpolation, **kwargs)

def image_resize_batch(x_batch: np.ndarray | list[np.ndarray], *args, **kwargs) -> np.ndarray:
    """resizes a bath of images to the given height and width"""
    fn = MemoryData if isinstance(x_batch[0], MemoryData) else np.asarray
    return fn([image_resize(x, *args, **kwargs) for x in x_batch])
