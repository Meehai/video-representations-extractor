"""
image_utils.py: generic utils for images manipulation. Repo at https://gitlab.com/meehai/image_utils.py.
Version: 2025.03.02.1
"""
from typing import NamedTuple
import os
from PIL import Image as PILImage, ImageDraw
import numpy as np
from loggez import make_logger

logger = make_logger("IMAGE_UTILS", exists_ok=True)

if os.getenv("TYPEGUARD", "0") == "1":
    from typeguard import install_import_hook
    install_import_hook(["image_utils", "image_utils_pil"])

try:
    import cv2
    DEFAULT_RESIZE_BACKEND = "cv2"
except ImportError:
    logger.error("OpenCV is not installed. Will use PIL for image_reisze")
    DEFAULT_RESIZE_BACKEND = "pil"

class PointIJ(NamedTuple):
    """defines a 2D point in IJ coordinates for images"""
    i: int
    j: int

Image = np.ndarray
Color = tuple[int, int, int]
Point2D = PointIJ | tuple[int, int] | tuple[float, float]
Shape = tuple[int, int, int]

# Module utilities

def _check_image(image: Image):
    assert image.dtype in (np.uint8, np.float32), f"{image.dtype=}"
    if image.dtype == np.uint8:
        assert image.shape[-1] in (3, 1), image.shape # RGB or grayscale only for now

def _scale(a: int, b: int, c: int) -> int:
    return int(b / a * c)

def _get_height_width(image_shape: Shape, height: int | None, width: int | None) -> tuple[int, int]:
    """used by image_resize to get height from width or vice-versa if one is missing while maintaining scale"""
    width = _scale(image_shape[0], height, image_shape[1]) if (width is None or width == -1) else width
    height = _scale(image_shape[1], width, image_shape[0]) if (height is None or height == -1) else height
    return height, width

def _get_px_from_perc(perc: float, image_shape: Shape) -> int:
    """returns the size in pixels from percents"""
    min_shape = perc * min(image_shape[0], image_shape[1]) / 100
    if min_shape < 1:
        logger.trace(f"{min_shape=} below 1 pixel. Returning 1")
    return max(1, int(min_shape))

def _check_points(p1: Point2D, p2: Point2D, image_shape: Shape) -> tuple[PointIJ, PointIJ]:
    p1, p2 = (p1, p2) if p1[0] < p2[0] else ((p1, p2) if p1[0] == p2[0] and p1[1] < p2[1] else (p2, p1))
    p1 = (min(p1[0], image_shape[0] - 1), min(p1[1], image_shape[1] - 1))
    p2 = (min(p2[0], image_shape[0] - 1), min(p2[1], image_shape[1] - 1))
    return PointIJ(*p1), PointIJ(*p2)

def _update(res: np.ndarray, us: np.ndarray, vs: np.ndarray, color: Shape):
    """safely write (sub-)pixels into an image without going out of bounds. 2.4 writes to both 2 and 3 position."""
    u_floor = us.astype(int).clip(0, res.shape[0] - 1)
    v_floor = vs.astype(int).clip(0, res.shape[1] - 1)
    u_ceil = np.ceil(us).astype(int).clip(0, res.shape[0] - 1)
    v_ceil = np.ceil(vs).astype(int).clip(0, res.shape[1] - 1)
    res[u_floor, v_floor] = color
    res[u_ceil, v_ceil] = color

# Public API

# Image manipulation functions (i.e. resizing).

def _image_resize_pil(image: Image, height: int, width: int, resample: PILImage.Resampling, **kwargs) -> Image:
    """image is supposed to be validated here, 3-sized rgb or grayscale with uint8 or float32 only"""
    out = np.empty((height, width, channels := image.shape[2]), dtype=image.dtype)
    if image.dtype == np.uint8:
        if image.shape[-1] == 1:
            image = image[..., 0] # grayscale pil uint8 needs to be 2D
        pil_image = PILImage.fromarray(image).resize((width, height), resample=resample, **kwargs)
        out[:] = np.asarray(pil_image).reshape(height, width, channels)
    elif image.dtype == np.float32:
        for c in range(channels):
            pil_image_c = PILImage.fromarray(image[..., c]).resize((width, height), resample=resample, **kwargs)
            out[..., c] = np.asarray(pil_image_c)
    return out

def _image_resize_cv2(image: Image, height: int, width: int, interpolation: int, **kwargs) -> Image:
    res = cv2.resize(image, dsize=(width, height), interpolation=interpolation, **kwargs)
    return res.astype(image.dtype).reshape(height, width, image.shape[2]) # for e.g. grayscale with (h, w, 1)

def image_resize(image: Image, height: int | None, width: int | None,
                 interpolation: str = "bilinear", backend: str = DEFAULT_RESIZE_BACKEND, **kwargs) -> Image:
    """Wrapper on top of Image(arr).resize((w, h), args) or cv2.resize. Sadly cv2 is faster so we cannot remove it."""
    # TODO: reimplement with image_utils primitives.
    _check_image(image)
    height, width = _get_height_width(image.shape, height, width)
    assert isinstance(height, int) and isinstance(width, int), (type(height), type(width))
    if image.shape[0:2] == (height, width):
        return image

    if backend.upper() == "CV2":
        interpolation_type = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "lanczos": cv2.INTER_LANCZOS4
        }[interpolation]
        res = _image_resize_cv2(image, height, width, interpolation=interpolation_type, **kwargs)
    elif backend.upper() == "PIL":
        interpolation_type: PILImage.Resampling = {
            "nearest": PILImage.Resampling.NEAREST,
            "bilinear": PILImage.Resampling.BILINEAR,
            "lanczos": PILImage.Resampling.LANCZOS,
        }[interpolation]
        res = _image_resize_pil(image, height, width, resample=interpolation_type, **kwargs)
    else:
        raise ValueError(str(backend))
    return res

def image_paste(image1: Image, image2: Image, top_left: Point2D=(0, 0),
                background_color: Color=(0, 0, 0), inplace: bool=False) -> Image:
    """Pastes two [0:255] images over each other. image  takes priority everywhere except where it's (0, 0, 0)"""
    _check_image(image1)
    _check_image(image2)
    top_left = PointIJ(*top_left)
    assert image1.shape[0] - top_left.i >= image2.shape[0], f"{image1.shape=}, {image2.shape=}, {top_left=}"
    assert image1.shape[1] - top_left.j >= image2.shape[1], f"{image1.shape=}, {image2.shape=}, {top_left=}"
    res = image1 if inplace else image1.copy()

    res_shifted = res[top_left.i:top_left.i + image2.shape[0], top_left.j:top_left.j + image2.shape[1]]
    mask: np.ndarray = (image2 == background_color).sum(-1, keepdims=True) == 3
    res_shifted[:] = res_shifted * mask + image2 * (~mask)
    return res

# Drawing functions

def image_draw_line(image: Image, p1: Point2D, p2: Point2D, color: Color,
                    thickness: float, inplace: bool=False) -> Image:
    """Draws a lines between two points with a given thickness"""
    _check_image(image)
    p1, p2 = _check_points(p1, p2, image.shape)
    assert p1 != p2, f"p1 and p2 cannot be the same: {p1=} {p2=}"
    thickness_px = _get_px_from_perc(thickness, image.shape)
    res = image if inplace else image.copy()

    m, b = 0, 0
    if p1.i != p2.i:
        m = (p1.j - p2.j) / (p1.i - p2.i)
        b = p1.j - m * p1.i

    if p1.i == p2.i: # horizontal
        vs = np.arange(p1.j, p2.j + 1).astype(int)
        us = vs * 0 + p1.i
        thickness_px = min(thickness_px, image.shape[0] - p1.i) # ensure we don't go out of border

        if thickness_px == 1:
            res[us, vs] = color
            return res

        for i in range(thickness_px):
            res[us + i - thickness_px // 2 + (thickness_px % 2 == 0), vs] = color

    elif p1.j == p2.j: # vertical line
        us = np.arange(p1.i, p2.i + 1).astype(int)
        vs = us * 0 + p1.j
        thickness_px = min(thickness_px, image.shape[1] - p1.j) # because they will shoot at us

        if thickness_px == 1:
            res[us, vs] = color
            return res

        for i in range(thickness_px):
            res[us, vs + i - thickness_px // 2 + (thickness_px % 2 == 0)] = color

    else: # diagonal line
        assert m != 0, f"{p1=}, {p2=}, {m=}, {b=}"

        skip_one = thickness_px == 2 # 3 lines: 1 top, 1 middle, 1 bot. Middle one is reduced by one to look nicer.
        us_mid = np.arange(p1.i + skip_one, p2.i + 1).astype(int)
        vs_mid = m * us_mid + b
        _update(res, us_mid, vs_mid, color)

        if thickness_px == 1:
            return res

        us = np.arange(p1.i, p2.i + 1 - skip_one)
        vs = m * us + b
        n = 1 + thickness_px // 2

        for i in range(1, n):
            _update(res, us - i // 2, vs + i - (i // 2), color) # top band starts to the right of middle
            _update(res, us + i // 2, vs - i + (i // 2), color) # bottom band starts one row below

    return res

def image_draw_rectangle(image: Image, top_left: Point2D, bottom_right: Point2D,
                         color: Color, thickness: float, inplace: bool=False) -> Image:
    """Draws a rectangle (i.e. bounding box) over an image. Thinkness is in percents w.r.t smallest axis (min 1)."""
    _check_image(image)
    top_left, bottom_right = PointIJ(*top_left), PointIJ(*bottom_right)

    if top_left.i > bottom_right.i:
        logger.trace(f"{top_left=}, {bottom_right=}. Swapping.")
        top_left, bottom_right = bottom_right, top_left
    res = image if inplace else image.copy()

    image_draw_line(res, p1=top_left, p2=(top_right := PointIJ(top_left.i, bottom_right.j)),
                    color=color, thickness=thickness, inplace=True)
    image_draw_line(res, top_right, bottom_right, color=color, thickness=thickness, inplace=True)
    image_draw_line(res, p1=bottom_right, p2=(bottom_left := PointIJ(bottom_right.i, top_left.j)),
                    color=color, thickness=thickness, inplace=True)
    image_draw_line(res, bottom_left, top_left, color=color, thickness=thickness, inplace=True)

    return res

def image_draw_polygon(image: Image, points: list[Point2D], color: Color, thickness: float,
                       inplace: bool=False) -> Image:
    """draws a polygon given some points"""
    _check_image(image)
    assert len(points) >= 2, "at least 2 points needed"
    points = [PointIJ(*p) for p in points]

    res = image if inplace else image.copy()
    for l, r in zip(points, [*points[1:], points[0]]): # noqa: E741
        image_draw_line(res, p1=l, p2=r, color=color, thickness=thickness, inplace=True)
    return res

def image_draw_circle(image: Image, center: Point2D, radius: float, color: Color, fill: bool,
                      outline_thickness: float | None = None, inplace: bool=False) -> Image:
    """draw a circle at a given center with a radius (in percents). Outline thickness is also in percents (or none)"""
    # TODO: reimplement with image_utils primitives.
    _check_image(image)
    img_pil = PILImage.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    r_px = _get_px_from_perc(radius, image.shape)
    assert (fill is True and outline_thickness is None) or not fill, "if fill is set, outline_thickness shouldn't be"
    outline_thickness_px = _get_px_from_perc(outline_thickness, image.shape) if outline_thickness is not None else 1
    center = PointIJ(*center)

    if fill:
        draw.ellipse((center.j - r_px, center.i - r_px, center.j + r_px, center.i + r_px), fill=color)
    else:
        draw.ellipse((center.j - r_px, center.i - r_px, center.j + r_px, center.i + r_px), outline=color,
                     width=outline_thickness_px)
    res = np.array(img_pil)
    if inplace:
        image[:] = res
    return res
