"""colorize semantic segmentation module -- based on the original M2F implementation but without matplotlib"""
import pycocotools.mask as mask_util
from skimage.draw import polygon # pylint: disable=no-name-in-module
from PIL import Image, ImageDraw
import numpy as np
from .pil_utils import _get_default_font, _pil_image_draw_textsize
from .image import image_blend
from .cv2_utils import cv2_findContours, cv2_RETR_CCOMP, cv2_CHAIN_APPROX_NONE, cv2_connectedComponentsWithStats

_AREA_THRESHOLD = 10
_LINE_WIDTH = 1.2
_LARGE_MASK_AREA_THRESH = 120_000
_WHITE = (255, 255, 255)

class _GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        assert isinstance(mask, np.ndarray), type(mask)
        assert mask.shape[1] != 2, mask.shape
        assert mask.shape == (height, width,), f"mask shape: {mask.shape}, target dims: {height}, {width}"
        self.mask = mask.astype(np.uint8)
        self.polygons, self.has_holes = self._mask_to_polygons(self.mask)

    def _mask_to_polygons(self, mask) -> tuple[list[tuple[int, int]], bool]:
        # cv2_RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2_CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2_findContours(mask.astype("uint8"), cv2_RETR_CCOMP, cv2_CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

def _add_text_with_background(image_array, text, position, font_size, color):
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)
    font = _get_default_font(font_size)

    text_width, text_height = _pil_image_draw_textsize(draw, text, font=font) # Calculate text size
    x, y = position # Define rectangle coordinates
    x = max(5, min(x, image_array.shape[1] - text_width - 5))
    y = max(5, min(y, image_array.shape[0] - text_height - 5))
    rectangle_coords = [x - 1, y + 2, x + text_width + 1, y + text_height]

    draw.rectangle(rectangle_coords, fill=(0, 0, 0)) # Draw black rectangle
    draw.text((x, y), text, fill=color, font=font) # add the text
    return np.array(image)

def _font_size_from_shape(shp) -> int:
    return max(13, min(20, min(*shp) // 20))

def _draw_text_in_mask(res: np.ndarray, binary_mask: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """
    Find proper places to draw text given a binary mask.
    """
    _num_cc, cc_labels, stats, _ = cv2_connectedComponentsWithStats(binary_mask, 8)
    if stats[1:, -1].size == 0:
        return res
    largest_component_id = np.argmax(stats[1:, -1]) + 1

    # draw text on the largest component, as well as other very large components.
    font_size_px =_font_size_from_shape(binary_mask.shape)
    for cid in range(1, _num_cc):
        if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
            center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1] # position based on median
            res = _add_text_with_background(res, text, center, font_size_px, color)
    return res

def _draw_polygon(vertices, shape, linewidth=1):
    """
    Draw a filled polygon with specified edge linewidth into a binary NumPy array, avoiding loops.

    :param vertices: A list or array of (x, y) tuples representing the polygon vertices.
    :param shape: A tuple (height, width) specifying the size of the output array.
    :param linewidth: The thickness of the polygon edges.
    :return: A binary NumPy array with the polygon drawn with filled interior and specified edge width.
    """
    vertices = np.array(vertices, dtype=np.float32)

    # Normalize linewidth as a radius
    radius = linewidth / 2.0

    # Calculate offsets for edge thickening
    edges = np.concatenate([vertices, vertices[:1]])  # Close the polygon
    edge_vectors = edges[1:] - edges[:-1]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
    normals = np.stack([-edge_vectors[:, 1], edge_vectors[:, 0]], axis=1)  # Perpendicular vector
    normals /= np.maximum(edge_lengths, 1e-6)  # Normalize to unit vectors

    # Expand polygon by the radius in both directions along normals
    expanded_vertices = []
    for direction in [-1, 1]:
        offset = direction * radius * normals
        expanded_vertices.append(edges[:-1] + offset)

    expanded_vertices = np.concatenate(expanded_vertices, axis=0)

    # Rasterize expanded polygon
    rr, cc = polygon(expanded_vertices[:, 1], expanded_vertices[:, 0], shape)
    binary_array = np.zeros(shape, dtype=bool)
    binary_array[rr, cc] = 1
    return binary_array

def colorize_sem_seg(sema: np.ndarray, rgb: np.ndarray | None, classes: list[str],
                     color_map: list[tuple[int, int, int]], alpha: float=0.8) -> np.ndarray:
    """colorize semantic segmentation -- based on Mask2Former's approach but without MPL"""
    classes = list(map(str, classes)) if all(isinstance(x, int) for x in classes) else classes
    assert all(isinstance(x, str) for x in classes), classes
    sema_rgb = np.zeros((*sema.shape, 3), dtype=np.uint8)
    labels, areas = np.unique(sema, return_counts=True)
    sorted_idxs = np.argsort(-areas).tolist()
    sorted_labels = labels[sorted_idxs]
    res = sema_rgb
    masks: list[_GenericMask] = []
    alpha_all_masks = np.full((*rgb.shape[0:2], 1), 0).astype(np.float32) # an array jost for alphas per mask
    polys = np.zeros(rgb.shape[0:2], dtype=bool) # polygons binary mask that is added at the end
    texts_data = [] # a list of all texts based on the class labels (if shown)
    for ix in sorted_idxs:
        masks.append(_GenericMask(sema == labels[ix], *sema.shape[0:2]))

    for ix, mask in enumerate(masks):
        res[mask.mask.astype(bool)] = color_map[sorted_labels[ix]] # draw mask
        if mask.has_holes:
            texts_data.append((mask.mask, classes[sorted_labels[ix]], _WHITE)) # texts are in fg
        else:
            for segment in mask.polygons:
                area = mask_util.area(mask_util.frPyObjects([segment], *sema.shape[0:2]))
                if area < _AREA_THRESHOLD:
                    continue
                texts_data.append((mask.mask, classes[sorted_labels[ix]], _WHITE)) # texts are in fg
                polys = polys | _draw_polygon(segment.reshape(-1, 2), sema.shape[0:2], linewidth=_LINE_WIDTH)

        if rgb is not None:
            alpha_all_masks[mask.mask.astype(bool)] = 1 - alpha

    # apply all the extra transformations: rgb+alpha overlap -> white polygons -> texts on top of the masks
    if rgb is not None:
        res = image_blend(res, rgb, alpha_all_masks)
    res[polys] = _WHITE
    for text_data in texts_data: # TODO: if texts overlap, show just the one with highest area maybe
        res = _draw_text_in_mask(res, *text_data)
    return res
