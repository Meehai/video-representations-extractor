"""image utils for VRE"""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps, ImageFont

from ..logger import logger
from .utils import get_closest_square, get_project_root

def _image_resize_cv2(data: np.ndarray, height: int, width: int, interpolation: str, **kwargs) -> np.ndarray:
    interpolation = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA,
                     "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[interpolation]
    _data = data if data.dtype == np.uint8 else data.astype(np.float32)
    return cv2.resize(_data, dsize=(width, height), interpolation=interpolation, **kwargs).astype(data.dtype)

def _image_resize_pil(data: np.ndarray, height: int, width: int, interpolation: str, **kwargs) -> np.ndarray:
    interpolation = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR}[interpolation]
    assert data.dtype == np.uint8, f"Only uint8 allowed, got {data.dtype}"
    pil_image = Image.fromarray(data).resize((width, height), interpolation, **kwargs)
    return np.asarray(pil_image)

def image_resize(data: np.ndarray, height: int, width: int, interpolation: str = "bilinear",
                 library: str = "cv2", **kwargs) -> np.ndarray:
    """image resize. Allows 2 libraries: PIL and opencv (to alleviate potential pre-trained issues)"""
    assert isinstance(height, int) and isinstance(width, int)
    return {"cv2": _image_resize_cv2, "PIL": _image_resize_pil}[library](data, height, width, interpolation, **kwargs)

def image_resize_batch(x_batch: np.ndarray | list[np.ndarray], *args, **kwargs) -> np.ndarray:
    """resizes a bath of images to the given height and width"""
    return np.array([image_resize(x, *args, **kwargs) for x in x_batch])

def image_write(x: np.ndarray, path: Path):
    """writes an image to a bytes string"""
    assert x.dtype == np.uint8, x.dtype
    res = cv2.imwrite(f"{path}", x[..., ::-1])
    assert res is not None, f"Image {x.shape} could not be saved to '{path}'"

def image_read(path: Path) -> np.ndarray:
    """Read an image from a path. Return uint8 [0:255] ndarray"""
    cv_res = cv2.imread(f"{path}")
    assert cv_res is not None, f"OpenCV returned None for '{path}'"
    bgr_image = cv_res[..., 0:3]
    b, g, r = cv2.split(bgr_image)
    image = cv2.merge([r, g, b]).astype(np.uint8)
    return image

def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """HSV to RGB color space conversion."""
    arr = hsv.astype(np.float64) / 255 if hsv.dtype == np.uint8 else hsv

    hi = np.floor(arr[..., 0] * 6)
    f = arr[..., 0] * 6 - hi
    p = arr[..., 2] * (1 - arr[..., 1])
    q = arr[..., 2] * (1 - f * arr[..., 1])
    t = arr[..., 2] * (1 - (1 - f) * arr[..., 1])
    v = arr[..., 2]

    out = np.choose(
        np.stack([hi, hi, hi], axis=-1).astype(np.uint8) % 6,
        np.stack([np.stack((v, t, p), axis=-1), np.stack((q, v, p), axis=-1), np.stack((p, v, t), axis=-1),
                  np.stack((p, q, v), axis=-1), np.stack((t, p, v), axis=-1), np.stack((v, p, q), axis=-1)]),
    )

    return out

def generate_diverse_colors(n: int, saturation: float, value: float) -> list[tuple[int, int, int]]:
    """generates a list of n diverse colors using the hue from the HSV transform"""
    assert 0 <= saturation <= 1, saturation
    assert 0 <= value <= 1, value
    colors = []
    for i in range(n):
        hue = i / n  # Vary the hue component
        rgb = hsv2rgb(np.array([hue, saturation, value]))
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

DEFAULT_FONT_HEIGHTS = {12: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 13: 4, 14: 5, 15: 7, 16: 8, 17: 10, 18: 11, 19: 12, 20: 14, 21: 15, 22: 17, 23: 18, 24: 19, 25: 21, 26: 22, 27: 24, 28: 25, 29: 26, 30: 28, 31: 29, 32: 31, 33: 32, 34: 33, 35: 35, 36: 36, 37: 38, 38: 39, 39: 40, 40: 42, 41: 44, 42: 45, 43: 46, 44: 48, 45: 49, 46: 51, 47: 52, 48: 53, 49: 55, 50: 56, 51: 58, 52: 59, 53: 60, 54: 62, 55: 63, 56: 65, 57: 66, 58: 67, 59: 69, 60: 70, 61: 72, 62: 73, 63: 74, 64: 76, 65: 77, 66: 79, 67: 80, 68: 81, 69: 83, 70: 84, 71: 86, 72: 87, 73: 88, 74: 90, 75: 91, 76: 93, 77: 94, 78: 95, 79: 97, 80: 98, 81: 100, 82: 101, 83: 102, 84: 104, 85: 105, 86: 107, 87: 108, 88: 109, 89: 111, 90: 112, 91: 114, 92: 115, 93: 116, 94: 118, 95: 119, 96: 121, 97: 122, 98: 123, 99: 125, 100: 126, 101: 128, 102: 129, 103: 130, 104: 132, 105: 133, 106: 135, 107: 136, 108: 137, 109: 139, 110: 140, 111: 142, 112: 143, 113: 144, 114: 146, 115: 147, 116: 149, 117: 150, 118: 151, 119: 153, 120: 154, 121: 156, 122: 157, 123: 158, 124: 160, 125: 161, 126: 163, 127: 164, 128: 165, 129: 167, 130: 168, 131: 170, 132: 171, 133: 172, 134: 174, 135: 175, 136: 177, 137: 178, 138: 179, 139: 181, 140: 182, 141: 184, 142: 185, 143: 186, 144: 188, 145: 189, 146: 191, 147: 192, 148: 194, 149: 195, 150: 196, 151: 198, 152: 199, 153: 201, 154: 202, 155: 203, 156: 205, 157: 206, 158: 208, 159: 209, 160: 210, 161: 212, 162: 213, 163: 215, 164: 216, 165: 217, 166: 219, 167: 220, 168: 222, 169: 223, 170: 224, 171: 226, 172: 227, 173: 229, 174: 230, 175: 231, 176: 233, 177: 234, 178: 236, 179: 237, 180: 238, 181: 240, 182: 241, 183: 243, 184: 244, 185: 245, 186: 247, 187: 248, 188: 250, 189: 251, 190: 252, 191: 254, 192: 255, 193: 257, 194: 258, 195: 259, 196: 261, 197: 262, 198: 264, 199: 265, 200: 266, 201: 268, 202: 269, 203: 271, 204: 272, 205: 273, 206: 275, 207: 276, 208: 278, 209: 279, 210: 280, 211: 282, 212: 283, 213: 285, 214: 286, 215: 287, 216: 289, 217: 290, 218: 292, 219: 293, 220: 294, 221: 296, 222: 297, 223: 299, 224: 300, 225: 301, 226: 303, 227: 304, 228: 306, 229: 307, 230: 308, 231: 310, 232: 311, 233: 313, 234: 314, 235: 315, 236: 317, 237: 318, 238: 320, 239: 321, 240: 322, 241: 324, 242: 325, 243: 327, 244: 328, 245: 329, 246: 331, 247: 332, 248: 334, 249: 335, 250: 336, 251: 338, 252: 339, 253: 341, 254: 342, 255: 343, 256: 345, 257: 346, 258: 348, 259: 349, 260: 350, 261: 352, 262: 353, 263: 355, 264: 356, 265: 357, 266: 359, 267: 360, 268: 362, 269: 363, 270: 364, 271: 366, 272: 367, 273: 369, 274: 370, 275: 371, 276: 373, 277: 374, 278: 376, 279: 377, 280: 378, 281: 380, 282: 381, 283: 383, 284: 384, 285: 385, 286: 387, 287: 388, 288: 390, 289: 391, 290: 392, 291: 394, 292: 395, 293: 397, 294: 398, 295: 399, 296: 401, 297: 402, 298: 404, 299: 405, 300: 406, 301: 408, 302: 409, 303: 411, 304: 412, 305: 413, 306: 415, 307: 416, 308: 418, 309: 419, 310: 420, 311: 422, 312: 423, 313: 425, 314: 426, 315: 427, 316: 429, 317: 430, 318: 432, 319: 433, 320: 434, 321: 436, 322: 437, 323: 439, 324: 440, 325: 441, 326: 443, 327: 444, 328: 446, 329: 447, 330: 448, 331: 450, 332: 451, 333: 453, 334: 454, 335: 455, 336: 457, 337: 458, 338: 460, 339: 461, 340: 462, 341: 464, 342: 465, 343: 467, 344: 468, 345: 469, 346: 471, 347: 472, 348: 474, 349: 475, 350: 476, 351: 478, 352: 479, 353: 481, 354: 482, 355: 483, 356: 485, 357: 486, 358: 488, 359: 489, 360: 490, 361: 492, 362: 493, 363: 495, 364: 496, 365: 497, 366: 499, 367: 500} # pylint: disable=line-too-long


def get_default_font(size_px: int | None = None):
    """
    Gets the default font for some pixel size. Downloads it if it's not provided in:
        '../../../resources/OpenSans-Bold.ttf'
    """
    if size_px is None:
        size_px = 12
    assert isinstance(size_px, int)
    global DEFAULT_FONT_HEIGHTS # pylint: disable=global-variable-not-assigned
    if size_px not in DEFAULT_FONT_HEIGHTS:
        logger.info(f"Size px {size_px} not in default_fonts")
        if size_px > max(DEFAULT_FONT_HEIGHTS.keys()):
            size_px = max(DEFAULT_FONT_HEIGHTS.keys())
        else:
            assert False
    font_path = get_project_root() / "resources/OpenSans-Bold.ttf"
    logger.debug2(f"Getting default font from '{font_path}' for desired height = '{size_px}' px")
    size = DEFAULT_FONT_HEIGHTS[size_px]
    if size == 0:
        logger.debug2("Asking for a size that is too small. Defaulting to font size = 1.")
        size = 1
    font = ImageFont.truetype(str(font_path), size=DEFAULT_FONT_HEIGHTS[size_px])
    return font

def image_add_text(image: np.ndarray, text: str, position: tuple[int, int], font: str | None = None,
                   font_size_px: int = None, font_color: str = "white") -> np.ndarray:
    """Adds a text to the image"""
    assert len(image.shape) == 3, f"Wrong image shape: {image.shape}"
    font = font if font is not None else get_default_font(font_size_px)
    height, width = image.shape[0:2]
    if not 0 <= position[0] <= height:
        logger.debug2(f"Height position ({position[0]}) is negative or outside boundaries. Hope you know what you do.")
    if not 0 <= position[1] <= width:
        logger.debug2(f"Width position ({position[1]}) is negative or outside boundaries. Hope you know what you do.")
    pil_image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil_image)
    # position (h, w) => draw.tetxt((w, h), ...)
    draw.text(position[::-1], text, font=font, fill=font_color)
    new_image = np.array(pil_image, dtype=image.dtype)
    return new_image

def _textsize(draw, text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def image_add_title(image: np.ndarray, text: str, font: str = None, font_color: str = "white", size_px: int = None,
                    background_color: str = "black", top_padding: int = None) -> np.ndarray:
    """Calls image_add_text to add title on an updated image with padding on top for space and text centered"""
    assert len(image.shape) == 3, f"Wrong image shape: {image.shape}"
    height, _ = image.shape[0: 2]
    pil_image = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(pil_image)

    if top_padding is None:
        top_padding = int(height * 0.15)
        logger.debug2(f"Top padding not provided. Giving 15% of the image = {top_padding}")

    if size_px is None:
        size_px = top_padding

    # Expand the image with (left=0, top=top_padding, right=0, bottom=0)
    border = (0, top_padding, 0, 0)
    expanded_image = ImageOps.expand(pil_image, border=border, fill=background_color)
    expanded_image = np.array(expanded_image)
    text_width, text_height = _textsize(draw, text, font)
    position = -text_height // 4.8, (expanded_image.shape[1] - text_width) // 2
    return image_add_text(expanded_image, text=text, position=position, font=font, font_color=font_color,
                          font_size_px=size_px)
