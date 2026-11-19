# Docs: build the site with opdoc (robosim) instead of build_docs.sh

**Created**: 2026-11-19
**Priority**: 2
**Closed**: n/a

docs/build_docs.sh is a hand port of what is now robosim's pkg/opdoc. opdoc does the same job (README home, prose pages, API, theme) plus file-relative link rewriting and a build that fails on broken links.

Blocker: opdoc has no way to exclude modules. VRE hides vendored model internals (*_impl, ultralytics, resources: 82 of 172 modules). Add an --exclude option to opdoc upstream first.

Then:
- vendor pkg/opdoc (pinned, see pkg/README.md); keep it out of VRE's installed packages (setup.py auto-discovers pkg/*)
- docs/source/{basics,examples}/*.md -> docs/source/<page>/PAGE.md with Menu Title/Position headers
- docs/build_docs.sh -> opdoc docs/ public --api-package vre --api-package vre_repository --import-path . --import-path pkg/image_utils --import-path pkg/vre-video; VRE CSS overrides -> docs/custom.css
- .gitlab-ci.yml pages job; pdoc pin as a docs extra in setup.py
