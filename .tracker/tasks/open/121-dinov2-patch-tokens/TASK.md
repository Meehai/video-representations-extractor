# DINOv2: patch tokens (pool: none) + explicit input_size

**Created**: 2026-11-16
**Priority**: 2
**Closed**: n/a

**Labels**: representation

## Description

Export DINOv2 patch tokens once and reuse them for dense matching (RoMa-style relative pose). The caller owns the rest
(resizing its images to the grid, fine CNN features, matching).

- `pool: none`: no pooling, `(h/14, w/14, D)` last-layer patch tokens (after the final layernorm), not normalized.
  `make_images`: top-3 PCA components as RGB (sign-fixed, per frame).
- `long_side` (aspect kept, rounded to 14) -> `input_size: (h, w)`, exact, multiples of 14 or `ValueError`. The grid is
  then known up front (e.g. 560x560 -> 40x40, as RoMa). Default `(378, 672)` = the old `long_side: 672` on 16:9.

Known caveats (not addressed here):
- Our position-embedding interpolation is transformers >= 4.45 (`size=`); RoMa's vendored DINOv2 uses the older
  `scale_factor` + 0.1 offset (~4% different features off 518). Check pose error once before relying on it.
- Disk: 40x40x1024 fp32 = 6.5MB/frame. fp16 storage would halve it (`load_from_disk` already upcasts).
- `input_size` is not stored next to the outputs (see 91).

## Done when

- [x] code (`dinov2.py`, `embedding_representation.py` docstring), on branch `dinov2-tokens`
- [x] unit + integration tests, e2e cfg, dinov2 README, `test/manual/dinov2/check_hf_parity.py`
- [ ] review gate green in CI
