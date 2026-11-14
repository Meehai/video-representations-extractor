# make_images should receive a ReprOut item, not assume .data exists

**Created**: 2024-12-23
**Priority**: 3
**Closed**: 2024-12-24

## Description

This function now is not pure (depends on external data: `.data` to exists).
This hinders the ability for example to make `vre_collage` parallel.
