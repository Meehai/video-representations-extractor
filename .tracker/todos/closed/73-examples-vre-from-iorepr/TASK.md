# Examples: add the VRE from IORepresentations code/notebook

**Created**: 2024-11-11
**Closed**: 2024-11-11
**Priority**: 3

## Description

- step 1): generate base dataset from the test video (i.e. 10 frames) [rgb/m2f_mapillary/m2f_coco/marigold/normals_marigold]
- step 2): generate iterative dataset from the base dataset [hsv/buildings]
- step 3): rerun step 2) and validate that no computation is done, just loading
