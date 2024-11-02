"""cv2 utils. All the calls to opencv must be condensed here"""
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
import requests

from .utils import get_project_root
from ..logger import vre_logger as logger

def pil_image_resize(data: np.ndarray, height: int, width: int, interpolation: str, **kwargs) -> np.ndarray:
    """Wrapper on top of Image(arr).resize((w, h), args)"""
    interpolation = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR}[interpolation]
    assert data.dtype == np.uint8, f"Only uint8 allowed, got {data.dtype}"
    pil_image = Image.fromarray(data).resize((width, height), interpolation, **kwargs)
    return np.asarray(pil_image)

def pil_image_add_title(image: np.ndarray, text: str, font: str = None, font_color: str = "white",
                        background_color: str = "black", top_padding: int = None, size_px: int = None) -> np.ndarray:
    """Calls image_add_text to add title on an updated image with padding on top for space and text centered"""

    def get_default_font(size_px: int | None = None):
        """
        Gets the default font for some pixel size. Downloads it if it's not provided in:
            '../../../resources/OpenSans-Bold.ttf'
        """
        default_font_heights = {12: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 13: 4, 14: 5, 15: 7, 16: 8, 17: 10, 18: 11, 19: 12, 20: 14, 21: 15, 22: 17, 23: 18, 24: 19, 25: 21, 26: 22, 27: 24, 28: 25, 29: 26, 30: 28, 31: 29, 32: 31, 33: 32, 34: 33, 35: 35, 36: 36, 37: 38, 38: 39, 39: 40, 40: 42, 41: 44, 42: 45, 43: 46, 44: 48, 45: 49, 46: 51, 47: 52, 48: 53, 49: 55, 50: 56, 51: 58, 52: 59, 53: 60, 54: 62, 55: 63, 56: 65, 57: 66, 58: 67, 59: 69, 60: 70, 61: 72, 62: 73, 63: 74, 64: 76, 65: 77, 66: 79, 67: 80, 68: 81, 69: 83, 70: 84, 71: 86, 72: 87, 73: 88, 74: 90, 75: 91, 76: 93, 77: 94, 78: 95, 79: 97, 80: 98, 81: 100, 82: 101, 83: 102, 84: 104, 85: 105, 86: 107, 87: 108, 88: 109, 89: 111, 90: 112, 91: 114, 92: 115, 93: 116, 94: 118, 95: 119, 96: 121, 97: 122, 98: 123, 99: 125, 100: 126, 101: 128, 102: 129, 103: 130, 104: 132, 105: 133, 106: 135, 107: 136, 108: 137, 109: 139, 110: 140, 111: 142, 112: 143, 113: 144, 114: 146, 115: 147, 116: 149, 117: 150, 118: 151, 119: 153, 120: 154, 121: 156, 122: 157, 123: 158, 124: 160, 125: 161, 126: 163, 127: 164, 128: 165, 129: 167, 130: 168, 131: 170, 132: 171, 133: 172, 134: 174, 135: 175, 136: 177, 137: 178, 138: 179, 139: 181, 140: 182, 141: 184, 142: 185, 143: 186, 144: 188, 145: 189, 146: 191, 147: 192, 148: 194, 149: 195, 150: 196, 151: 198, 152: 199, 153: 201, 154: 202, 155: 203, 156: 205, 157: 206, 158: 208, 159: 209, 160: 210, 161: 212, 162: 213, 163: 215, 164: 216, 165: 217, 166: 219, 167: 220, 168: 222, 169: 223, 170: 224, 171: 226, 172: 227, 173: 229, 174: 230, 175: 231, 176: 233, 177: 234, 178: 236, 179: 237, 180: 238, 181: 240, 182: 241, 183: 243, 184: 244, 185: 245, 186: 247, 187: 248, 188: 250, 189: 251, 190: 252, 191: 254, 192: 255, 193: 257, 194: 258, 195: 259, 196: 261, 197: 262, 198: 264, 199: 265, 200: 266, 201: 268, 202: 269, 203: 271, 204: 272, 205: 273, 206: 275, 207: 276, 208: 278, 209: 279, 210: 280, 211: 282, 212: 283, 213: 285, 214: 286, 215: 287, 216: 289, 217: 290, 218: 292, 219: 293, 220: 294, 221: 296, 222: 297, 223: 299, 224: 300, 225: 301, 226: 303, 227: 304, 228: 306, 229: 307, 230: 308, 231: 310, 232: 311, 233: 313, 234: 314, 235: 315, 236: 317, 237: 318, 238: 320, 239: 321, 240: 322, 241: 324, 242: 325, 243: 327, 244: 328, 245: 329, 246: 331, 247: 332, 248: 334, 249: 335, 250: 336, 251: 338, 252: 339, 253: 341, 254: 342, 255: 343, 256: 345, 257: 346, 258: 348, 259: 349, 260: 350, 261: 352, 262: 353, 263: 355, 264: 356, 265: 357, 266: 359, 267: 360, 268: 362, 269: 363, 270: 364, 271: 366, 272: 367, 273: 369, 274: 370, 275: 371, 276: 373, 277: 374, 278: 376, 279: 377, 280: 378, 281: 380, 282: 381, 283: 383, 284: 384, 285: 385, 286: 387, 287: 388, 288: 390, 289: 391, 290: 392, 291: 394, 292: 395, 293: 397, 294: 398, 295: 399, 296: 401, 297: 402, 298: 404, 299: 405, 300: 406, 301: 408, 302: 409, 303: 411, 304: 412, 305: 413, 306: 415, 307: 416, 308: 418, 309: 419, 310: 420, 311: 422, 312: 423, 313: 425, 314: 426, 315: 427, 316: 429, 317: 430, 318: 432, 319: 433, 320: 434, 321: 436, 322: 437, 323: 439, 324: 440, 325: 441, 326: 443, 327: 444, 328: 446, 329: 447, 330: 448, 331: 450, 332: 451, 333: 453, 334: 454, 335: 455, 336: 457, 337: 458, 338: 460, 339: 461, 340: 462, 341: 464, 342: 465, 343: 467, 344: 468, 345: 469, 346: 471, 347: 472, 348: 474, 349: 475, 350: 476, 351: 478, 352: 479, 353: 481, 354: 482, 355: 483, 356: 485, 357: 486, 358: 488, 359: 489, 360: 490, 361: 492, 362: 493, 363: 495, 364: 496, 365: 497, 366: 499, 367: 500} # pylint: disable=line-too-long
        if size_px is None:
            size_px = 12
        assert isinstance(size_px, int)
        if size_px not in default_font_heights:
            logger.info(f"Size px {size_px} not in default_fonts")
            if size_px > max(default_font_heights.keys()):
                size_px = max(default_font_heights.keys())
            else:
                assert False
        font_path = get_project_root() / "resources/OpenSans-Bold.ttf"
        if not font_path.exists():
            font_path.parent.mkdir(exist_ok=True, parents=True)
            with open(font_path, "wb") as file:
                file.write(requests.get("https://github.com/edx/edx-fonts/raw/refs/heads/master/open-sans/fonts/Bold/OpenSans-Bold.ttf").content) # pylint: disable=all
        logger.debug2(f"Getting default font from '{font_path}' for desired height = '{size_px}' px")
        size = default_font_heights[size_px]
        if size == 0:
            logger.debug2("Asking for a size that is too small. Defaulting to font size = 1.")
            size = 1
        font = ImageFont.truetype(str(font_path), size=default_font_heights[size_px])
        return font

    def image_add_text(image: np.ndarray, text: str, position: tuple[int, int], font: str | None = None,
                    font_size_px: int = None, font_color: str = "white") -> np.ndarray:
        """Adds a text to the image"""
        assert len(image.shape) == 3, f"Wrong image shape: {image.shape}"
        font = font if font is not None else get_default_font(font_size_px)
        height, width = image.shape[0:2]
        if not 0 <= position[0] <= height:
            logger.debug2(f"Height position ({position[0]}) is negative or outside boundaries ({height=}).")
        if not 0 <= position[1] <= width:
            logger.debug2(f"Width position ({position[1]}) is negative or outside boundaries ({width=}).")
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

    assert len(image.shape) == 3, f"Wrong image shape: {image.shape}"
    height, _ = image.shape[0: 2]
    pil_image = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(pil_image)

    if top_padding is None:
        top_padding = int(height * 0.15)
        logger.debug2(f"Top padding not provided. Giving 15% of the image = {top_padding}")

    if size_px is None:
        size_px = top_padding

    if font is None:
        font = get_default_font(size_px=size_px)

    # Expand the image with (left=0, top=top_padding, right=0, bottom=0)
    border = (0, top_padding, 0, 0)
    expanded_image = ImageOps.expand(pil_image, border=border, fill=background_color)
    expanded_image = np.array(expanded_image)
    text_width, text_height = _textsize(draw, text, font)
    position = -text_height // 4.8, (expanded_image.shape[1] - text_width) // 2
    return image_add_text(expanded_image, text=text, position=position, font=font, font_color=font_color)


def pil_image_read(path: str) -> np.ndarray:
    """PIL image reader"""
    # TODO: for grayscale, this returns a RGB image too
    img_pil = Image.open(path)
    img_np = np.array(img_pil, dtype=np.uint8)
    # grayscale -> 3 gray channels repeated.
    if img_pil.mode == "L":
        return np.repeat(img_np[..., None], 3, axis=-1)
    # RGB or RGBA
    return img_np[..., 0:3]

def pil_image_write(file: np.ndarray, path: str):
    """PIL image writer"""
    assert file.min() >= 0 and file.max() <= 255
    img = Image.fromarray(file.astype(np.uint8), "RGB")
    img.save(path)
