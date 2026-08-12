"""
image_utils_pil.py - PIL-based compatible API for compatibility and tests.
The aim is to be as close as reasonable possible to it without going crazy (i.e. no periodic functions on lines lol)
"""
import numpy as np
from PIL import Image, ImageDraw

from image_utils import PointIJ, Color, _check_image, _get_px_from_perc, logger, _check_points

def image_draw_line_pil(image: np.ndarray, p1: PointIJ, p2: PointIJ, color: Color,
                        thickness: float, inplace: bool=False) -> np.ndarray:
    """Draws a lines between two points with a given thickness"""
    _check_image(image)
    # PIL inconsistency (TODO failure case): (w=(20, 50), p1=(5, 25), p2=(5, 10), thck=20); p1-p2 vs p2-p1.
    p1, p2 = _check_points(p1, p2, image.shape)

    thickness_px = _get_px_from_perc(thickness, image.shape)

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.line((p1.j, p1.i, p2.j, p2.i), fill=color, width=thickness_px)
    res = np.array(img_pil)
    if inplace:
        image[:] = res

    return res

def image_draw_rectangle_pil(image: np.ndarray, top_left: PointIJ, bottom_right: PointIJ,
                             color: Color, thickness: float, inplace: bool=False) -> np.ndarray:
    """Draws a rectangle (i.e. bounding box) over an image. Thinkness is in percents w.r.t smallest axis (min 1)."""
    _check_image(image)
    top_left, bottom_right = PointIJ(*top_left), PointIJ(*bottom_right)

    if top_left.i > bottom_right.i:
        logger.trace(f"{top_left=}, {bottom_right=}. Swapping.")
        top_left, bottom_right = bottom_right, top_left

    thickness_px = _get_px_from_perc(thickness, image.shape)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([top_left.j, top_left.i, bottom_right.j, bottom_right.i], outline=color, width=thickness_px)
    res = np.array(img_pil)
    if inplace:
        image[:] = res
    return res

def image_draw_polygon_pil(image: np.ndarray, points: list[PointIJ], color: Color, thickness: int,
                           inplace: bool=False) -> np.ndarray:
    """draws a polygon given some points"""
    _check_image(image)
    assert len(points) >= 2, "at least 2 points needed"
    points = [PointIJ(*p) for p in points]
    thickness_px = _get_px_from_perc(thickness, image.shape)

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    for l, r in zip(points, [*points[1:], points[0]]): # noqa: E741
        l, r = _check_points(l, r, image.shape)
        draw.line((l.j, l.i, r.j, r.i), fill=color, width=thickness_px)
    res = np.array(img_pil)
    if inplace:
        image[:] = res
    return res
