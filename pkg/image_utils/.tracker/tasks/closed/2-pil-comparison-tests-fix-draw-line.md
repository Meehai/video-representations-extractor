# Pil comparison tests and fix image_draw_line

**Created**: 2025-12-05
**Closed**: 2025-12-05
**Priority**: 2

**MR**: !2 (merged) · branch `pil_comparison_tests` · commit `b8e3299`
<https://gitlab.com/meehai/image_utils.py/-/merge_requests/2>

Imported from GitLab. Original MR description was empty; summary from the merge diff.

## What shipped
- Bug fixes to native `image_draw_line`.
- New `test/unit/image_draw_vs_pil_test.py` — compares our raster against PIL.
- Helpers added to `image_utils_pil.py` (the PIL reference backend).
- Notebook + `image_draw_test.py` updated.
