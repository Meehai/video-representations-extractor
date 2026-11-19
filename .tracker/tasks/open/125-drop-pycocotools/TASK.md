# Drop pycocotools: numpy COCO polygon area in the semseg colorizer

**Created**: 2026-11-19
**Priority**: 2
**Closed**: n/a

**Labels**: deps

## Description

pycocotools (C extension) was a `vre_repository` dep for one call: the semseg colorizer's
`mask_util.area(mask_util.frPyObjects([poly], h, w))`, which drops polygons below `_AREA_THRESHOLD` before drawing
outlines and labels. Replace it with `_coco_polygon_area`, a numpy port of maskApi.c `rleFrPoly` + `rleArea` for one
polygon, and remove pycocotools from `setup.py`. No vendoring: one ~35-line function, not a package.

Cost: ~70-85x slower per polygon than C (Python loop per edge; ~1 ms per 100-vertex contour). On the stored e2e
semantic frames that is ~24 ms/frame on a 50-1200 ms render; worst case (per-pixel noise maps, 1000+ polygons)
~1.5 s/frame. Vectorizing the upsample step would close most of the gap if it ever matters.

## Done when

- [x] `_coco_polygon_area` in `vre_repository/utils/colorizer/colorize_semantic_segmentation.py`; pycocotools out
      of `setup.py`
- [x] exact parity with pycocotools 2.0.10 (`test/vre_repository/integration/parity/test_parity_colorizer.py`;
      skipped where pycocotools isn't installed). Also checked once: 455k fuzzed polygons (1080p/4K contours,
      borders, .5 ties, huge coords, h=1/w=1) with 0 mismatches; 153/153 colorizer renders byte-identical
- [ ] review gate green in CI
