# Vendor overrides: robosim's shim replaces pip overrides

**Created**: 2026-11-19
**Priority**: 2
**Closed**: 2026-11-19

**Labels**: deps

## Description

The pip `overrides==7.7.0` core dep is replaced by robosim's first-party `@overrides` shim, vendored at
`pkg/overrides` (upstream `robosim/pkg/overrides`, tracked in robosim). Same API. At class-definition time it checks
that the method exists in a base class and that its signature matches the base's as written; opt out with
`@overrides(check_signature=False)`.

The signature check found overrides that did not match their bases; fixed to match: `resize(...) -> ReprOut`
(`Representation`, `ResizableRepresentationMixin`), `compute(..., dep_data: list[ReprOut] | None = None)` (optical
flow and semantic segmentation bases), `Mask2Former.vre_setup(load_weights: bool)`,
`SoftSegmentationRepresentation.n_channels -> int`. vre-video is bumped to `1e04e9e` for its own (`shape`, `fps`
return types).

## Done when

- [x] code applied (!259): `pkg/overrides`, pip `overrides` out of `setup.py`, the signature fixes, vre-video bump,
      `pkg/README.md` + AGENTS.md vendored-packages table, `--import-path pkg/overrides` in `docs/build_docs.sh`
- [x] full pytest gate green locally (159 passed, 1 xpassed)
- [ ] review gate green in CI (!259)
