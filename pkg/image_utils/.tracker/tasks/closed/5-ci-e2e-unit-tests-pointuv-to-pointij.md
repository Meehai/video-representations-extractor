# add CI: e2e/unit tests and refactor PointUV in PointIJ

**Created**: 2025-12-24
**Closed**: 2025-12-24
**Priority**: 2

**MR**: !5 (merged) · branch `PointIJ_e2e` · commit `5c10b33`
<https://gitlab.com/meehai/image_utils.py/-/merge_requests/5>

Imported from GitLab. Original MR description was empty; summary from the merge diff.

## What shipped
- `.gitlab-ci.yml`: linters + unit tests + e2e (`draw_sun_on_cat`) jobs.
- Renamed `PointUV` → `PointIJ` across `image_utils.py` / `image_utils_pil.py` and callers
  (coords are IJ, not XY).
- `examples/minipaint/main.py` and tests updated to the new name.
