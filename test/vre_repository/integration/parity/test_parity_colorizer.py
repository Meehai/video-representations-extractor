"""
The semantic segmentation colorizer's COCO polygon area (numpy port of maskApi.c rleFrPoly + rleArea) vs pycocotools:
same polygon, same pixel count. The colorizer drops polygons below an area threshold, so any off-by-one changes which
outlines and labels get drawn. Runs only if pycocotools is installed.
"""
import numpy as np
import pytest
from vre_repository.utils.colorizer.colorize_semantic_segmentation import _coco_polygon_area, _GenericMask

mask_util = pytest.importorskip("pycocotools.mask")

def test_parity_coco_polygon_area_mask_contours_vs_pycocotools():
    """the real input: cv2 contours (+0.5) of the per-class masks of a blocky label map"""
    rng = np.random.default_rng(42)
    for _ in range(100):
        h, w = (int(x) for x in rng.integers(8, 160, size=2))
        sema = rng.integers(0, 5, size=(h // 8 + 1, w // 8 + 1)).repeat(8, axis=0).repeat(8, axis=1)[0:h, 0:w]
        for label in np.unique(sema):
            for poly in _GenericMask(sema == label, h, w).polygons:
                expected = int(mask_util.area(mask_util.frPyObjects([poly], h, w))[0])
                assert _coco_polygon_area(poly, h, w) == expected, (h, w, poly.tolist())

def test_parity_coco_polygon_area_random_floats_vs_pycocotools():
    """arbitrary float vertices, self-intersecting and partly outside the image"""
    rng = np.random.default_rng(42)
    for _ in range(3000):
        h, w = (int(x) for x in rng.integers(1, 200, size=2))
        poly = rng.uniform(-10, [w + 10, h + 10], size=(rng.integers(3, 40), 2)).flatten()
        expected = int(mask_util.area(mask_util.frPyObjects([poly], h, w))[0])
        assert _coco_polygon_area(poly, h, w) == expected, (h, w, poly.tolist())

def test_parity_coco_polygon_area_degenerate_vs_pycocotools():
    """repeated vertices (zero-length edges), integer vertices and 1-pixel-wide images"""
    rng = np.random.default_rng(42)
    for _ in range(3000):
        h, w = (int(x) for x in rng.integers(1, 30, size=2))
        poly = rng.integers(0, [w + 1, h + 1], size=(rng.integers(2, 10), 2)).repeat(2, axis=0).flatten().astype(float)
        expected = int(mask_util.area(mask_util.frPyObjects([poly], h, w))[0])
        assert _coco_polygon_area(poly, h, w) == expected, (h, w, poly.tolist())
