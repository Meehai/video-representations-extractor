# Split vre in two: vre[core] and vre[repository].

**Created**: 2025-11-24
**Closed**: 2026-02-16
**Priority**: 3

## Description

Better idea! we should split VRE in two (requirements/pip wise):

- `vre[core]` -> lightweight preferably only python stuff + numpy + loggez + overrides (the usual suspects pretty much)
- `vre[repository]` -> heavyweight with torch and all other external deps, including cv2.

For external projects, like [neo-transformers](https://gitlab.com/video-representations-extractor/neo-transformers) we should use `vre[core]` only
