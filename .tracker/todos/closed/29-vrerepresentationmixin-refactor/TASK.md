# refactor: VRERepresentationMixin

**Created**: 2023-11-13
**Closed**: 2023-11-14
**Priority**: 3
**Labels**: refactor

## Description

- `VRERepresentationMixin`: adds `vre_setup` (Exists already), `vre_make(video, ix: slice, make_images: bool)`, `vre_dependencies -> list[ReprOutput]` methods to `Representation`
- `make` and `make_images` get frames as inputs
- `self.video` is no longer in representation's attributes
- vre loop logic is simplified
- represnetations are now decoupled from VRE and only interact through the Mixin's class method.
