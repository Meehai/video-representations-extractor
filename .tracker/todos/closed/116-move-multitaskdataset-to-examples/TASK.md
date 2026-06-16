# Remove MultiTaskDataset from vre and move it to examples.

**Created**: 2026-02-16
**Closed**: 2026-03-04
**Priority**: 3

## Description

A few CI/tests depend on it, so we need to find a way to decouple that. Then, we move the implementation in examples and hint the users that this is a way (out of many others) to interact with the exported .npz files.
