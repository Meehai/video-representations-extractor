"""
image_utils_pil.py - PIL-based compatible API for compatibility and tests.
The aim is to be as close as reasonable possible to it without going crazy (i.e. no periodic functions on lines lol)
"""
from typing import NamedTuple
import os
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageOps
from loggez import make_logger

if os.getenv("TYPEGUARD", "0") == "1":
    from typeguard import install_import_hook
    install_import_hook(["image_utils", "image_utils_pil"])

logger = make_logger("IMAGE_UTILS", exists_ok=True)

class PointIJ(NamedTuple):
    """defines a 2D point in IJ coordinates for images"""
    i: int
    j: int

Image = np.ndarray
Color = tuple[int, int, int]
Point2D = PointIJ | tuple[int, int] | tuple[float, float]
Shape = tuple[int, int, int]

def _get_px_from_perc(perc: float, image_shape: Shape) -> int:
    """returns the size in pixels from percents"""
    min_shape = perc * min(image_shape[0], image_shape[1]) / 100
    if min_shape < 1:
        logger.trace(f"{min_shape=} below 1 pixel. Returning 1")
    return max(1, int(min_shape))

def _check_points(p1: Point2D, p2: Point2D, image_shape: Shape) -> tuple[PointIJ, PointIJ]:
    p1, p2 = (p1, p2) if p1[0] < p2[0] else ((p1, p2) if p1[0] == p2[0] and p1[1] < p2[1] else (p2, p1))
    p1_ = (min(p1[0], image_shape[0] - 1), min(p1[1], image_shape[1] - 1))
    p2_ = (min(p2[0], image_shape[0] - 1), min(p2[1], image_shape[1] - 1))
    return PointIJ(int(p1_[0]), int(p1_[1])), PointIJ(int(p2_[0]), int(p2_[1]))

def _check_image(image: Image):
    assert image.dtype in (np.uint8, np.float32), f"{image.dtype=}"
    assert image.shape[-1] in (3, 1), image.shape # RGB or grayscale only for now


def image_draw_line(image: Image, p1: Point2D, p2: Point2D, color: Color,
                    thickness: float, inplace: bool=False) -> Image:
    """Draws a lines between two points with a given thickness"""
    _check_image(image)
    # PIL inconsistency (TODO failure case): (w=(20, 50), p1=(5, 25), p2=(5, 10), thck=20); p1-p2 vs p2-p1.
    p1, p2 = _check_points(p1, p2, image.shape)

    thickness_px = _get_px_from_perc(thickness, image.shape)

    img_pil = PILImage.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.line((p1.j, p1.i, p2.j, p2.i), fill=color, width=thickness_px)
    res = np.array(img_pil)
    if inplace:
        image[:] = res

    return res

def image_draw_rectangle(image: Image, top_left: Point2D, bottom_right: Point2D,
                         color: Color, thickness: float, inplace: bool=False) -> Image:
    """Draws a rectangle (i.e. bounding box) over an image. Thinkness is in percents w.r.t smallest axis (min 1)."""
    _check_image(image)
    top_left, bottom_right = PointIJ(*top_left), PointIJ(*bottom_right)

    if top_left.i > bottom_right.i:
        logger.trace(f"{top_left=}, {bottom_right=}. Swapping.")
        top_left, bottom_right = bottom_right, top_left

    thickness_px = _get_px_from_perc(thickness, image.shape)
    img_pil = PILImage.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([top_left.j, top_left.i, bottom_right.j, bottom_right.i], outline=color, width=thickness_px)
    res = np.array(img_pil)
    if inplace:
        image[:] = res
    return res

def image_draw_polygon(image: Image, points: list[Point2D], color: Color, thickness: float,
                       inplace: bool=False) -> Image:
    """draws a polygon given some points"""
    _check_image(image)
    assert len(points) >= 2, "at least 2 points needed"
    points = [PointIJ(*p) for p in points]
    thickness_px = _get_px_from_perc(thickness, image.shape)

    img_pil = PILImage.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    for l, r in zip(points, [*points[1:], points[0]]): # noqa: E741
        l, r = _check_points(l, r, image.shape)
        draw.line((l.j, l.i, r.j, r.i), fill=color, width=thickness_px)
    res = np.array(img_pil)
    if inplace:
        image[:] = res
    return res

def image_draw_circle(image: Image, center: Point2D, radius: float, color: Color,
                      fill: bool, outline_thickness: int | None = None, inplace: bool=False) -> Image:
    """draw a circle at a given center with a radius (in percents). Outline thickness is also in percents (or none)"""
    _check_image(image)
    img_pil = PILImage.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    r_px = _get_px_from_perc(radius, image.shape)
    assert (fill is True and outline_thickness is None) or not fill, "if fill is set, outline_thickness shouldn't be"
    outline_thickness_px = 1 if outline_thickness is None else _get_px_from_perc(outline_thickness, image.shape)
    center = PointIJ(float(center[0]), float(center[1]))

    if fill:
        draw.ellipse((center.j - r_px, center.i - r_px, center.j + r_px, center.i + r_px), fill=color)
    else:
        draw.ellipse((center.j - r_px, center.i - r_px, center.j + r_px, center.i + r_px), outline=color,
                     width=outline_thickness_px)
    res = np.array(img_pil)
    if inplace:
        image[:] = res
    return res

def image_add_title(image: Image, text: str, position_ij: Point2D, font_size: float,
                    font_color: str = "white", top_padding: int | None = None,
                    top_padding_color: str = "black") -> Image:
    """Adds a title to an image. Optionally can add a top padding (expand image) so we don't write on top of it"""
    assert len(image.shape) == 3, f"Wrong image shape: {image.shape}"
    pil_image = PILImage.fromarray(image.astype(np.uint8))

    top_padding = top_padding or 0
    if top_padding > 0:
        border = (0, top_padding, 0, 0)
        pil_image = ImageOps.expand(pil_image, border=border, fill=top_padding_color)

    draw = ImageDraw.Draw(pil_image)
    # position (h, w) => draw.tetxt((w, h), ...)
    draw.text(position_ij[::-1], text, fill=font_color, font_size=font_size)
    new_image = np.array(pil_image, dtype=image.dtype)

    return new_image
