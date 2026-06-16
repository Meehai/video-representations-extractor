# split vre/utils in two: vre specific and vre repository specific

**Created**: 2026-02-18
**Closed**: 2026-03-02
**Priority**: 3

## Description

we should be able to install vre[core] without any issues and replicate vre_repository stuff only if needed.

For example VRE shouldn't care about image stuff. The only exception is resizing, for which we can have a pure numpy resize (or torch?).
