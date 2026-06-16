# Handle resolution inconsistencies when loading from disk

**Created**: 2024-10-07
**Priority**: 2
**Labels**: feature, refactor

## Description

we need to treat corner cases better: if depth_dpt was computed at (384, 672) and stored as resized to (720, 1080) (video frame shape), then we should load it back to (384, 672). We can store the resize metadata in extra for all the reprs.
