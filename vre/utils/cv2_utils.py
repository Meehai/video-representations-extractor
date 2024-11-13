"""cv2 utils. All the calls to opencv must be condensed here so eventually we can get rid of them"""
# pylint: disable=invalid-name, no-member
from pathlib import Path
import numpy as np
from ..logger import vre_logger as logger

try:
    import cv2
    cv2_CHAIN_APPROX_SIMPLE = cv2.CHAIN_APPROX_SIMPLE
    cv2_CHAIN_APPROX_NONE = cv2.CHAIN_APPROX_NONE
    cv2_RETR_TREE = cv2.RETR_TREE
    cv2_RETR_EXTERNAL = cv2.RETR_EXTERNAL
    cv2_RETR_CCOMP = cv2.RETR_CCOMP
    cv2_MORPH_OPEN = cv2.MORPH_OPEN
    cv2_MORPH_CLOSE = cv2.MORPH_CLOSE
    cv2_COLOR_RGB2BGR = cv2.COLOR_RGB2BGR
    cv2_COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    cv2_BORDER_CONSTANT = cv2.BORDER_CONSTANT
    cv2_LINE_AA = cv2.LINE_AA
    cv2_INTER_LINEAR = cv2.INTER_LINEAR
    cv2_INTER_NEAREST = cv2.INTER_NEAREST
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
    assert res is not None and res is not False, f"Image {x.shape} could not be saved to '{path}'"

def cv2_image_resize(data: np.ndarray, height: int, width: int, interpolation: str, **kwargs) -> np.ndarray:
    """wraper on top of cv2.image_resize"""
    interpolation = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA,
                     "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[interpolation]
    _data = data if data.dtype == np.uint8 else data.astype(np.float32)
    return cv2.resize(_data, dsize=(width, height), interpolation=interpolation, **kwargs).astype(data.dtype)

def cv2_connectedComponents(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.connectedComponents(*args, **kwargs)

def cv2_Canny(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.Canny(*args, **kwargs)

def cv2_findContours(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.findContours(*args, **kwargs)

def cv2_boundingRect(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.boundingRect(*args, **kwargs)

def cv2_drawContours(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.drawContours(*args, **kwargs)

def cv2_morphologyEx(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.morphologyEx(*args, **kwargs)

def cv2_cvtColor(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.cvtColor(*args, **kwargs)

def cv2_connectedComponentsWithStats(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.connectedComponentsWithStats(*args, **kwargs)

def cv2_copyMakeBorder(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.copyMakeBorder(*args, **kwargs)

def cv2_getTextSize(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.getTextSize(*args, **kwargs)

def cv2_rectangle(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.rectangle(*args, **kwargs)

def cv2_putText(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.putText(*args, **kwargs)

def cv2_transform(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.transform(*args, **kwargs)

def cv2_warpAffine(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.warpAffine(*args, **kwargs)

def cv2_getRotationMatrix2D(*args, **kwargs):
    """Wrapper on top of cv2"""
    return cv2.getRotationMatrix2D(*args, **kwargs)
