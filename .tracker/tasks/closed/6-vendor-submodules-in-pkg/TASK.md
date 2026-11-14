# Vendor submodules in pkg/, drop requirements files, drop Dockerfile

**Created**: 2026-08-12
**Priority**: 2
**Closed**: 2026-11-14

## Description

Align packaging with robosim's conventions:

- Replaced git submodules (image_utils, vre-video) with in-tree vendored copies under `pkg/`,
  committed as-is at the exact pinned commits (image_utils at `370728c` — its later commits have
  breaking changes not yet fixed in VRE, do NOT bump blindly; vre-video at `8c9e5b5`).
  Removed `.gitmodules`, the gitlinks, and `.git/modules`.
- `setup.py` is now the single source of truth for deps: auto-discovers `pkg/<repo>/` packages
  (one layout, `pkg/<repo>/<package>/`; flat upstream image_utils is copied into `pkg/image_utils/image_utils/`),
  `install_requires` = core, `extras_require["repository"]` = heavy deps. Deleted
  `requirements.txt` and `requirements-extra.txt`.
- Removed `Dockerfile` (out of scope).
- `.gitignore`: dropped the stale `image_utils/` entry; vendored content force-added so the
  broad `*.png`/`resources/` patterns don't drop tracked vendored assets.
- `.gitlab-ci.yml`: no `GIT_SUBMODULE_STRATEGY`, no `-r requirements.txt`; `pip install -e .`
  for core, CPU-only torch pre-install then `pip install -e ".[repository]"`; pylint excludes `pkg/`.
- `docs/build_docs.sh` PYTHONPATH now points at `pkg/` (docs already built with pdoc).
- Added `pkg/README.md` documenting the vendored packages + pinned commits.

Verified: `pip install -e .` in a clean venv, `import vre` works, `test/vre` (54 tests) passes,
`docs/build_docs.sh` builds.
