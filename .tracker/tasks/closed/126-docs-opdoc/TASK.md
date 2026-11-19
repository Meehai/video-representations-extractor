# Docs: build the site with opdoc (robosim) instead of build_docs.sh

**Created**: 2026-11-19
**Priority**: 2
**Closed**: 2026-11-19

The hand-ported pdoc script is replaced by opdoc (pip, `opdoc==0.2.3`, `docs` extra in setup.py).
`docs/build_docs.sh` stays as the one place holding the opdoc flags:

    opdoc docs/ public --api-package vre --api-package vre_repository \
      --import-path . --import-path pkg/image_utils --import-path pkg/vre-video \
      --exclude '*_impl' --exclude '*ultralytics' --exclude '*.resources'

- prose pages: `docs/source/<page>/PAGE.md` (Menu Title/Position header); page URLs unchanged
- links are file-relative (as on GitLab); a broken one fails the build
- no `docs/custom.css`: opdoc's base style already carries VRE's old overrides
