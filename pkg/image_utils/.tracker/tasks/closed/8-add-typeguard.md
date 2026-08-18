# add typeguard

**Created**: 2026-07-31
**Priority**: 2

**MR**: !8 (open) · branch `typeguard` (current) · commit `8ea9343`
<https://gitlab.com/meehai/image_utils.py/-/merge_requests/8>

Imported from GitLab. Original MR description was empty; summary from the branch diff vs master.

## What's in it
- Runtime type checking via `typeguard`, gated behind `TYPEGUARD=1`.
- Hooks in `image_utils.py` and `image_utils_pil.py`.
- CI (`.gitlab-ci.yml`) installs `typeguard` and runs tests with `TYPEGUARD=1`.
- New `test/e2e/run_all.sh` runner (exports `TYPEGUARD=1`, runs `draw_sun_on_cat`).

## Bugs found (2026-07-31)
- **Unscoped hook.** `install_import_hook()` with no args → `packages=None` → typeguard
  instruments *every* module incl. PIL. PIL's runtime annotations use TYPE_CHECKING-only names
  (`StrOrBytesPath`) → `NameError` in `PIL/ImageFile.py`. This is the "errors in PIL".
  Fix: scope to `["image_utils", "image_utils_pil"]` (one list covers both import modes).
- **Hook too late.** Installed *inside* the modules → runs after they're compiled → our own code
  was never checked (`TYPEGUARD=1 pytest test/unit` was a false green: 35 "passed", 0 checked).
  Fix: install the hook *before* importing, test-side.
- **CI typo.** `pip typeguard` (no `install`) + e2e job never set `TYPEGUARD=1`. Both fixed.

## Done (tester — test-side wiring)
- `test/conftest.py` — scoped hook, before any test import. Verified: our code now type-checked.
- `test/e2e/draw_sun_on_cat/main.py` — same scoped hook at top, before imports.
- `.gitlab-ci.yml` — `pip install typeguard` + `TYPEGUARD=1` on the e2e job.
- `test/manual/typeguard/instrumentation_check.py` — guard proving the hook instruments our code.

## Resolved (dev, commits 4488be2 / 866e8a8 / 8ea9343)
- **Hooks scoped in place** (not deleted): both files now call
  `install_import_hook(["image_utils", "image_utils_pil"])`. PIL/numpy no longer instrumented.
  Redundant with `test/conftest.py` but harmless. Note: an in-module hook can't instrument its
  own module (too late) — `image_utils.py`'s own funcs are checked only via `test/conftest.py`.
- **Annotations widened**: `Point2D = PointIJ | tuple[int, int] | tuple[float, float]` and
  `Shape = tuple[int, int, int]`, applied across point/shape params. typeguard now accepts the
  plain-tuple callers.
- Correction: an earlier note here claimed "55 findings" — that was **stale `.pyc`**
  (`opt-typeguard451`) from the pre-widening source. Clear `__pycache__` when toggling `TYPEGUARD`.

Current: clean cache → `TYPEGUARD=1` **58 passed** (unit+integration), e2e **Score 1.0 OK**.

## Docs (same branch)
- `.docs/build_api_reference.py` generates the README API table (public funcs of both files,
  backend = numpy/PIL/both). Idempotent, marker-delimited. Re-run after adding/renaming a func.

## Done when
- [x] Hooks scoped; annotations fixed; `TYPEGUARD=1` green on unit + integration + e2e.
- [ ] MR !8 merged into `master`.
