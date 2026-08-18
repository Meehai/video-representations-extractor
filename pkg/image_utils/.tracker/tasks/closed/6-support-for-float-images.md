# Support for float images

**Created**: 2026-03-01
**Closed**: 2026-03-01
**Priority**: 2

**MR**: !6 (merged) · branch `support-for-float-images` · commit `f6cad67`
<https://gitlab.com/meehai/image_utils.py/-/merge_requests/6>

Imported from GitLab. Original MR description was empty; summary from the merge diff.

## What shipped
- `image_utils.py` accepts float images, not just `uint8`.
- New `test/unit/image_resize_test.py` coverage for float paths.
- README note + e2e `main.py` tweak; `image_draw_vs_pil_test.py` moved into place.
