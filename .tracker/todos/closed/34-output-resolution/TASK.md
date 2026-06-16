# output resolution

**Created**: 2024-04-26
**Closed**: 2024-05-08
**Priority**: 3

## Description

Add `--output_resolution` to vre and a new method to representation: `resize(outputs: RepresentationOutput) -> RepresentationOutput`.

Useful if we run a network on a bigger video for better resolution, but we want to store the results in a lower res.
