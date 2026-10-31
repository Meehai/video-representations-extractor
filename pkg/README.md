# Vendored packages

This directory holds external packages that we use in VRE, pinned to an exact commit and committed
in-tree (no `.git`), instead of git submodules. This is the "vendor directory" pattern: a clone works
offline, there is no submodule init dance, and CI has no `GIT_SUBMODULE_STRATEGY`. We bump a package by
copying its files here at the new commit (check `setup.py` -- the packages under `pkg/` are
auto-discovered and put on the import path, so no per-package wiring is needed).

One layout: `pkg/<repo>/<package>/` (`pkg/vre-video/vre_video/`, `pkg/image_utils/image_utils/`).
`image_utils` upstream is flat (repo root = package), so its three `.py` files are copied into
`pkg/image_utils/image_utils/` when bumping. Its vendored `test/` tree imports the flat modules
(`import image_utils_pil`), so run it with `PYTHONPATH=pkg/image_utils/image_utils` if ever needed.

Vendored packages (pinned commits):
- [image_utils](./image_utils) - image helpers (resize, paste, drawing). Pinned to `370728c`
  on purpose: later commits add breaking changes (typeguard instrumentation, new API) not yet
  fixed in VRE -- do NOT bump blindly.
- [vre-video](./vre-video) - VRE's video reading/writing library (`import vre_video`).

CAVEAT: vendoring puts a package's code on the path but NOT its third-party deps. Each package's own
`setup.py`/`requirements.txt` here is informational; the deps we actually need are declared in the
top-level `setup.py` (the single source of truth). Vendored `test/` trees are not installed.
