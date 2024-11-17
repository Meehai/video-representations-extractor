"""image utils for VRE"""
from pathlib import Path
import numpy as np

from ..logger import vre_logger as logger
from .utils import get_closest_square
from .repr_memory_layout import MemoryData
from .cv2_utils import cv2_image_resize, cv2_image_write, cv2_image_read
from .pil_utils import pil_image_resize, pil_image_add_title, pil_image_read, pil_image_write

def image_resize(data: np.ndarray, height: int | None, width: int | None, interpolation: str = "bilinear",
                 library: str = "cv2", **kwargs) -> np.ndarray:
    """image resize. Allows 2 libraries: PIL and opencv (to alleviate potential pre-trained issues)"""
    assert ((width is None) or width == -1) + ((height is None) or height == -1) <= 1, "At least one must be set"
    def _scale(a: int, b: int, c: int) -> int:
        return int(a / b * c)
    width = _scale(data.shape[0], height, data.shape[1]) if (width is None or width == -1) else width
    height = _scale(data.shape[1], width, data.shape[0]) if (height is None or height == -1) else height
    assert isinstance(height, int) and isinstance(width, int), (type(height), type(width))
    return {"cv2": cv2_image_resize, "PIL": pil_image_resize}[library](data, height, width, interpolation, **kwargs)

def image_resize_batch(x_batch: np.ndarray | list[np.ndarray], *args, **kwargs) -> np.ndarray:
    """resizes a bath of images to the given height and width"""
    fn = MemoryData if isinstance(x_batch[0], MemoryData) else np.asarray
    return fn([image_resize(x, *args, **kwargs) for x in x_batch])

def image_write(x: np.ndarray, path: Path, library: str = "PIL"):
    """writes an image to a bytes string"""
    assert x.dtype == np.uint8, x.dtype
    return {"cv2": cv2_image_write, "PIL": pil_image_write}[library](x, path)

def image_read(path: Path, library: str = "cv2") -> np.ndarray:
    """Read an image from a path. Return uint8 [0:255] ndarray"""
    return {"cv2": cv2_image_read, "PIL": pil_image_read}[library](path)

def collage_fn(images: list[np.ndarray], rows_cols: tuple[int, int] = None, pad_bottom: int = 0,
               pad_right: int = 0, titles: list[str] = None, pad_to_max: bool = False, **title_kwargs) -> np.ndarray:
    """
    Make a concatenated collage based on the desired r,c format
    Parameters:
    - images A stack of images
    - rows_cols Tuple for number of rows and columns
    - pad_bottom An integer to pad the images on top, only valid in rows [2: n_rows]. TODO: what is this measured in?
    - pad_right An integer to pad images on right, only valid on columns [2: n_cols]. TODO: what is this measured in?
    - titles Titles for each image. Optional.
    - pad_to_max If True, pad all images to the max size of all images. If False, all image must be the same shape.

    Return: A numpy array of stacked images according to (rows, cols) inputs.
    """
    def _pad_to_max(imgs: list[np.ndarray]) -> list[np.ndarray]:
        """pad all images to the max shape of the list"""
        max_h = max(img.shape[0] for img in imgs)
        max_w = max(img.shape[1] for img in imgs)
        assert all(img.shape[2] == imgs[0].shape[2] for img in imgs)

        if all(img.shape == imgs[0].shape for img in imgs):
            return imgs

        logger.debug(f"Padding images to fit max size: {max_h}x{max_w}")
        res = []
        for img in imgs:
            new_img = np.pad(img, ((0, max_h - img.shape[0]), (0, max_w - img.shape[1]), (0, 0)), constant_values=255)
            res.append(new_img)
        return res

    assert len(images) > 1, "Must give at least two images to the collage"
    if rows_cols is None:
        rows_cols = get_closest_square(len(images))
        logger.debug2(f"row_cols was not set. Setting automatically to {rows_cols} based on number of images")
    assert len(rows_cols) == 2, f"rows_cols must be a tuple with 2 numbers, got: {rows_cols}"
    if np.prod(rows_cols) > len(images):
        logger.debug2(f"rows_cols: {rows_cols} greater than n images: {len(images)}. Padding with black images!")
    assert not all(x is None for x in images), "All images are None"

    if pad_to_max:
        images = _pad_to_max(images)

    shapes = [x.shape for x in [img for img in images if img is not None]]

    # np.pad uses [(0, 0), (0, 0), (0, 0)] to pad (a, b) on each channge of H,W,C. Our images may be H,W or H,W,C
    # If they are H, W, C then we care about [(0, pad_bottom), (0, pad_right), (0, 0)]
    pad = np.zeros((len(shapes[0]), 2), dtype=int)
    pad[0, 1] = pad_bottom
    pad[1, 1] = pad_right

    if any(x is None for x in images):
        logger.debug("Some images are None. Padding with black images!")
        images = [np.zeros(shapes[0], dtype=np.uint8) if x is None else x for x in images]
        shapes = [x.shape for x in images]

    if pad.sum() != 0:
        images = [np.pad(image, pad) for image in images]
        shapes = [x.shape for x in images]

    if titles is not None:
        images = [image_add_title(image, title, **title_kwargs) for (image, title) in zip(images, titles)]
        shapes = [x.shape for x in images]

    if np.std(shapes, axis=0).sum() != 0:
        raise ValueError(f"Shapes not equal: {shapes}. Use pad_to_max=True to pad images to max shape.")

    # Put all the results in a new array
    result = np.zeros((rows_cols[0] * rows_cols[1], *shapes[0]), dtype=np.uint8)
    result[0: len(images)] = np.array(images)
    result = result.reshape((rows_cols[0], rows_cols[1], *shapes[0]))
    result = np.concatenate(np.concatenate(result, axis=1), axis=1)
    # remove pad right from last image
    if pad_right != 0:
        result = result[:, 0: result.shape[1] - pad_right]
    if pad_bottom != 0:
        result = result[0: result.shape[0] - pad_bottom]
    return result

def image_add_title(image: np.ndarray, text: str, font: str = None, font_color: str = "white", size_px: int = None,
                    background_color: str = "black", top_padding: int = None, library: str = "PIL") -> np.ndarray:
    """Calls image_add_text to add title on an updated image with padding on top for space and text centered"""
    return {"PIL": pil_image_add_title}[library](image=image, text=text, font=font, font_color=font_color,
                                                 size_px=size_px, background_color=background_color,
                                                 top_padding=top_padding)

def image_add_border(image: np.ndarray, color: tuple[int, int, int] | int, thicc: int | None,
                     add_x: bool = False, inplace: bool = False) -> np.ndarray:
    """
    Given an image, add rectangles to it on each side. Optionally, cross it with an X (1 line to each diagonal).
    The original image is not altered unless inplace is set to True.
    Parameters:
    - image The image that is bordered.
    - color The color of the border. Must be a [0:255] tuple.
    """
    assert len(image.shape) == 3, f"Wrong image shape: {image.shape}"
    assert thicc is None or (isinstance(thicc, int) and thicc > 0), thicc
    color = [color, color, color] if isinstance(color, int) else color
    assert len(color) == 3, f"Wrong color shape: {color}"
    h, w = image.shape[0: 2]
    if thicc is None:
        logger.debug2(f"Thicc not provided, defaulting to {thicc}, based on {h=} and {w=}.")
        thicc = max(1, min(7, h // 33)) if h >= w else max(1, min(7, w // 33)) # heuristic, to be changed if it's bad.

    idx = np.linspace(0, 1, max(h, w))
    x = np.arange(w) if h >= w else np.round(idx * (w - 1)).astype(int)
    y = np.arange(h) if w >= h else np.round(idx * (h - 1)).astype(int)

    new_image = image if inplace else image.copy()
    if add_x:
        for t in range(thicc):
            new_image[np.clip(y + t, 0, h - 1), np.clip(x, 0, w - 1)] = color
            new_image[np.clip(h - (y + 1 + t), 0, h - 1), np.clip(x, 0, w - 1)] = color
    new_image[0: thicc] = color
    new_image[-thicc:] = color
    new_image[:, 0: thicc] = color
    new_image[:, -thicc:] = color
    return new_image
