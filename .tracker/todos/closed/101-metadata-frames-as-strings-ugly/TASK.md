# a bit ugly that in RepresentationMetadata frames are stored as strings

**Created**: 2025-02-25
**Closed**: 2025-03-10
**Priority**: 3

## Description

This is due to how json.dump/load works, but it adds a lot of `map(str, frames)` to our code to maintain compatibility.

Perhaps we should just do this on load/store manually and ensure that the frames in `run_stats` are integer always.

PS: we are **not** going to do named frames in this library **ever** to not complicate things.
