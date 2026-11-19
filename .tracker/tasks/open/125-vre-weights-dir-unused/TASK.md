# VRE_WEIGHTS_DIR is documented and exported in CI, but nothing reads it

**Created**: 2026-11-19
**Priority**: 3
**Closed**: n/a

**Labels**: bug, docs

## Description

`docs/source/basics/architecture-and-design.md` says `VRE_WEIGHTS_DIR` is where weights are downloaded, and
`.gitlab-ci.yml` exports it. No code reads it: `vre_repository/utils/weights_repository.py::fetch_weights` always
downloads into the representation's own `vre_repository/<repr>/weights/`, replacing the git-lfs pointer.

Decide one: implement it (fetch into `$VRE_WEIGHTS_DIR/<repr>/...` when set; CI's weights cache path follows), or drop
it from the docs and CI. Today the docs promise a feature that doesn't exist.
