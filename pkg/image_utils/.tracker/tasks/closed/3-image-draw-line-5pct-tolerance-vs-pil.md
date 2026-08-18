# image_draw_line with 5% tolerance compared to PIL. Big improvements to the testing area.

**Created**: 2025-12-07
**Closed**: 2025-12-07
**Priority**: 2

**MR**: !3 (merged) · branch `more-bug-fixes-todraw-line` · commit `2567ae8`
<https://gitlab.com/meehai/image_utils.py/-/merge_requests/3>

Imported from GitLab. Original MR description was empty; summary from the merge diff.

## What shipped
- `image_draw_line` now matches PIL within 5% tolerance.
- First e2e test: `test/e2e/draw_sun_on_cat/` (`main.py`, `run.sh`, `image_compare.py`).
- Expanded `image_draw_vs_pil_test.py` and `image_utils_test.py`.
- More PIL reference helpers in `image_utils_pil.py`.
