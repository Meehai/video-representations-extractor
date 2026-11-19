# Remove cv2

**Created**: 2023-11-01
**Priority**: 3
**Closed**: n/a

**Labels**: deps

## Description

Drop `opencv-python` from `vre_repository` (`REQUIRED_REPOSITORY` in `setup.py`). The calls go through
`vre_repository/vre_image_utils/cv2_utils.py`; the one exception is `cv2.resize` in the vendored `image_utils`. Per
call: use image_utils if it has it, else PIL (and an image_utils upstream task), else a numpy port in `cv2_utils.py`
(same name and signature as cv2), proven against cv2 by a parity test in `test/vre_repository/integration/parity/`
(skipped where cv2 isn't installed).

## Done when

Done (!259):
- [x] Canny: `vre_repository/edges/canny.py` is a numpy port of `cv2.Canny`, bit-exact. Checked once: 13k fuzzed
      images (noise, blocks, exact ties; 1x1 to 120x120; grey/RGB/RGBA; apertures 3/5/7; L1/L2; any thresholds;
      cv2 multi- and single-threaded) and 144 full-size photo configs, 0 mismatches; `test_parity_canny.py` keeps
      a slice. Cost: ~175 ms per 1080p RGB frame vs ~3 ms for cv2 (22 threads), ~42 ms at 540p. `cv2_Canny` removed.
- [x] SIFT (already numpy) checked vs `cv2.SIFT_create()` in `test_parity_sift.py`: ~99.7% of the keypoints match
      both ways (pt and size within 0.01, angle within 0.5 deg), descriptors within 3/255. The misses are
      orientation near-ties from cv2's approximate atan2.

Live calls left (port + parity test):
- [ ] `cv2.resize` in `pkg/image_utils`: the default resize backend when cv2 imports, behind
      `vre.utils.image_resize_batch` (the hot path). Upstream change; `image_utils` is pinned on purpose.
- [x] semseg colorizer (`vre_repository/utils/colorizer/colorize_semantic_segmentation.py`): `findContours`
      (`RETR_CCOMP`, `CHAIN_APPROX_NONE`) and `connectedComponentsWithStats` are numpy ports in `cv2_utils.py`
      (Suzuki-Abe border following as in opencv; run-length union-find with opencv's label order), bit-exact in
      `test_parity_cv2.py`. Same colorizer output and time on a 540x960 map.
- [x] FastSAM `plot_to_result(withContours=True)` (`soft_segmentation/fastsam/fastsam_impl/prompt.py`):
      `findContours` (`RETR_TREE`, `CHAIN_APPROX_SIMPLE`) is the numpy port; `drawContours` is
      `image_utils.image_draw_polygon` (2px, not pixel-equal to cv2's thick lines).

Opt-in or dead code left (delete):
- [x] `cv2_image_read` / `cv2_image_write` (`imread`, `imwrite`, `split`, `merge`): removed; callers use PIL
      (`Image.open` / `Image.fromarray(...).save`) directly.
- [x] FastSAM paths the representation never reaches, ported anyway: `_get_bbox_from_mask` is a numpy bbox and
      `_crop_image` a channel flip; `LetterBox` pads with PIL `ImageOps.expand` (image_utils task: constant-colour
      pad); `Annotator.box_label` draws with `image_utils.image_draw_rectangle` + `image_add_title`, text size from
      PIL (image_utils task: text size).
- [x] unused `cv2_utils.py` wrappers: `connectedComponents`, `morphologyEx`, `transform`, `warpAffine`,
      `getRotationMatrix2D`, `cv2_image_resize`.

Then:
- [x] `opencv-python` out of `setup.py`, cv2 out of the docs' dep lists
      (`docs/source/introduction-and-installation/PAGE.md`, `docs/source/vre-repository/PAGE.md`).
