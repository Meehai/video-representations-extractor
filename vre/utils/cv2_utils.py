"""cv2 utils. All the calls to opencv must be condensed here so eventually we can get rid of them"""
from pathlib import Path
import numpy as np
from ..logger import vre_logger as logger

try:
    import cv2
except ImportError as e:
    logger.warning(f"Cannot import cv2: {e}")

def cv2_image_read(path: Path) -> np.ndarray:
    """wrapper on top of cv2.imread"""
    cv_res = cv2.imread(f"{path}")
    assert cv_res is not None, f"OpenCV returned None for '{path}'"
    bgr_image = cv_res[..., 0:3]
    b, g, r = cv2.split(bgr_image)
    image = cv2.merge([r, g, b]).astype(np.uint8)
    return image

def cv2_image_write(x: np.ndarray, path: Path):
    """wrapper on top of cv2.imwrite"""
    assert x.dtype == np.uint8, x.dtype
    res = cv2.imwrite(f"{path}", x[..., ::-1])
    assert res is not None, f"Image {x.shape} could not be saved to '{path}'"

def cv2_image_resize(data: np.ndarray, height: int, width: int, interpolation: str, **kwargs) -> np.ndarray:
    """wraper on top of cv2.image_resize"""
    interpolation = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA,
                     "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[interpolation]
    _data = data if data.dtype == np.uint8 else data.astype(np.float32)
    return cv2.resize(_data, dsize=(width, height), interpolation=interpolation, **kwargs).astype(data.dtype)
