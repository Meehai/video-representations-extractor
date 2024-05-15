"""image utils for VRE"""
from pathlib import Path
import numpy as np
from skimage.transform import resize
from skimage.io import imsave, imread
from skimage.color import hsv2rgb

from ..logger import logger
from .utils import get_closest_square

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

def collage_fn(images: list[np.ndarray], rows_cols: tuple[int, int] = None, pad_bottom: int = 0,
               pad_right: int = 0, titles: list[str] = None, pad_to_max: bool = False) -> np.ndarray:
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

    assert titles is None, "Titles cannot be set in VRE. Use media-processing-lib for this"

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
