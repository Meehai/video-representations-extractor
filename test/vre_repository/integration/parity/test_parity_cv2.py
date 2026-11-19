"""
The numpy ports of opencv (vre_repository.vre_image_utils.cv2_utils, plus the FastSAM bbox and letterbox) vs cv2: same
input, same output. Bit-exact: contours, hierarchy, labels and stats are integers and the centroids are integer sums
divided by the area, so any mismatch is a real behaviour difference. Runs only if opencv is installed.
"""
import numpy as np
from PIL import Image
import pytest
from vre.utils import get_project_root
from vre_repository.vre_image_utils.cv2_utils import (cv2_findContours, cv2_connectedComponentsWithStats,
                                                      cv2_RETR_CCOMP, cv2_RETR_TREE, cv2_CHAIN_APPROX_NONE,
                                                      cv2_CHAIN_APPROX_SIMPLE)
from vre_repository.soft_segmentation.fastsam.fastsam_impl.prompt import FastSAMPrompt
from vre_repository.soft_segmentation.fastsam.fastsam_impl.results import LetterBox

cv2 = pytest.importorskip("cv2")

@pytest.mark.parametrize("mode", [cv2_RETR_CCOMP, cv2_RETR_TREE])
@pytest.mark.parametrize("method", [cv2_CHAIN_APPROX_NONE, cv2_CHAIN_APPROX_SIMPLE])
def test_parity_find_contours_random_vs_cv2(mode: int, method: int):
    """blocky masks: holes, islands in holes, isolated pixels, image-edge contact; 1x1 to 59x59; any non-zero value"""
    rng = np.random.default_rng(42)
    for _ in range(500):
        h, w, block = (int(x) for x in rng.integers(1, [60, 60, 6]))
        mask = rng.random((h // block + 1, w // block + 1)) < rng.uniform(0.1, 0.9)
        mask = (mask.repeat(block, 0).repeat(block, 1)[0:h, 0:w] * rng.integers(1, 256)).astype(np.uint8)
        contours, hierarchy = cv2_findContours(mask, mode, method)
        expected_contours, expected_hierarchy = cv2.findContours(mask.copy(), mode, method)
        assert len(contours) == len(expected_contours), (h, w, block)
        assert all(np.array_equal(a, b) for a, b in zip(contours, expected_contours)), (h, w, block)
        assert (hierarchy is None and expected_hierarchy is None) or np.array_equal(hierarchy, expected_hierarchy)

@pytest.mark.parametrize("mode, method", [(cv2_RETR_CCOMP, cv2_CHAIN_APPROX_NONE),
                                          (cv2_RETR_TREE, cv2_CHAIN_APPROX_SIMPLE)])
def test_parity_find_contours_large_vs_cv2(mode: int, method: int):
    """540x960 noise: thousands of contours (cv2 stores border ids in 7 bits, so ids wrap) and long nested borders"""
    rng = np.random.default_rng(42)
    for block in [1, 3, 16]:
        mask = rng.random((540 // block + 1, 960 // block + 1)) < 0.5
        mask = mask.repeat(block, 0).repeat(block, 1)[0:540, 0:960].astype(np.uint8)
        contours, hierarchy = cv2_findContours(mask, mode, method)
        expected_contours, expected_hierarchy = cv2.findContours(mask.copy(), mode, method)
        assert len(contours) == len(expected_contours) > 127, (block, len(contours), len(expected_contours))
        assert all(np.array_equal(a, b) for a, b in zip(contours, expected_contours)), block
        assert np.array_equal(hierarchy, expected_hierarchy), block

@pytest.mark.parametrize("name", ["demo1.jpg", "demo2.jpg"])
def test_parity_find_contours_photos_vs_cv2(name: str):
    """thresholded photos: natural shapes at full size"""
    grey = np.array(Image.open(get_project_root() / "resources" / name).convert("L"))
    for threshold in [64, 128, 192]:
        mask = (grey > threshold).astype(np.uint8)
        for mode, method in [(cv2_RETR_CCOMP, cv2_CHAIN_APPROX_NONE), (cv2_RETR_TREE, cv2_CHAIN_APPROX_SIMPLE)]:
            contours, hierarchy = cv2_findContours(mask, mode, method)
            expected_contours, expected_hierarchy = cv2.findContours(mask.copy(), mode, method)
            assert len(contours) == len(expected_contours), (threshold, mode, method)
            assert all(np.array_equal(a, b) for a, b in zip(contours, expected_contours)), (threshold, mode, method)
            assert np.array_equal(hierarchy, expected_hierarchy), (threshold, mode, method)

def test_parity_find_contours_empty_and_full_vs_cv2():
    """no foreground: no contours, no hierarchy. All foreground: the image border is one contour"""
    assert cv2_findContours(np.zeros((5, 7), np.uint8), cv2_RETR_CCOMP, cv2_CHAIN_APPROX_NONE) == ((), None)
    assert cv2.findContours(np.zeros((5, 7), np.uint8), cv2_RETR_CCOMP, cv2_CHAIN_APPROX_NONE) == ((), None)
    contours, hierarchy = cv2_findContours(np.ones((5, 7), np.uint8), cv2_RETR_TREE, cv2_CHAIN_APPROX_SIMPLE)
    expected_contours, expected_hierarchy = cv2.findContours(np.ones((5, 7), np.uint8), cv2.RETR_TREE,
                                                             cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == len(expected_contours) == 1 and np.array_equal(contours[0], expected_contours[0])
    assert np.array_equal(hierarchy, expected_hierarchy)

@pytest.mark.parametrize("connectivity", [4, 8])
def test_parity_connected_components_with_stats_random_vs_cv2(connectivity: int):
    """blocky masks, 1x1 to 59x59, including all-foreground (empty background label) and all-background"""
    rng = np.random.default_rng(42)
    for _ in range(500):
        h, w, block = (int(x) for x in rng.integers(1, [60, 60, 6]))
        mask = rng.random((h // block + 1, w // block + 1)) < rng.uniform(0, 1)
        mask = mask.repeat(block, 0).repeat(block, 1)[0:h, 0:w].astype(np.uint8)
        n, labels, stats, centroids = cv2_connectedComponentsWithStats(mask, connectivity)
        expected_n, expected_labels, expected_stats, expected_centroids = \
            cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
        assert n == expected_n and np.array_equal(labels, expected_labels), (h, w, block)
        assert np.array_equal(stats, expected_stats), (h, w, block)
        assert np.array_equal(centroids, expected_centroids, equal_nan=True), (h, w, block)

@pytest.mark.parametrize("connectivity", [4, 8])
def test_parity_connected_components_with_stats_large_vs_cv2(connectivity: int):
    """540x960 noise: thousands of components; cv2 labels such images in parallel stripes"""
    rng = np.random.default_rng(42)
    for block in [1, 2, 7]:
        mask = rng.random((540 // block + 1, 960 // block + 1)) < 0.5
        mask = mask.repeat(block, 0).repeat(block, 1)[0:540, 0:960].astype(np.uint8)
        n, labels, stats, centroids = cv2_connectedComponentsWithStats(mask, connectivity)
        expected_n, expected_labels, expected_stats, expected_centroids = \
            cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
        assert n == expected_n and np.array_equal(labels, expected_labels), block
        assert np.array_equal(stats, expected_stats) and np.array_equal(centroids, expected_centroids), block

def test_parity_fastsam_bbox_from_mask_vs_cv2():
    """FastSAM's mask bbox (numpy) vs the union of cv2's outer contours' bounding rects it replaced"""
    rng = np.random.default_rng(42)
    for _ in range(500):
        mask = rng.random((int(rng.integers(1, 40)), int(rng.integers(1, 40)))) < rng.uniform(0.01, 0.5)
        mask[rng.integers(0, mask.shape[0]), rng.integers(0, mask.shape[1])] = True
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = np.array([cv2.boundingRect(contour) for contour in contours])
        expected = [rects[:, 0].min(), rects[:, 1].min(), (rects[:, 0] + rects[:, 2]).max(),
                    (rects[:, 1] + rects[:, 3]).max()]
        assert FastSAMPrompt._get_bbox_from_mask(None, mask) == expected, mask.shape

@pytest.mark.parametrize("shape", [(100, 180), (180, 100), (64, 64), (33, 257)])
def test_parity_fastsam_letterbox_vs_cv2(shape: tuple[int, int]):
    """FastSAM's letterbox (PIL border) vs cv2.resize + cv2.copyMakeBorder with the same padding"""
    image = np.random.default_rng(42).integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    res = LetterBox((128, 256), auto=False)(image=image)
    r = min(128 / shape[0], 256 / shape[1])
    new_w, new_h = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (256 - new_w) / 2, (128 - new_h) / 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (new_h, new_w) != shape else image
    expected = cv2.copyMakeBorder(resized, int(round(dh - 0.1)), int(round(dh + 0.1)), int(round(dw - 0.1)),
                                  int(round(dw + 0.1)), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    assert np.array_equal(res, expected), shape
