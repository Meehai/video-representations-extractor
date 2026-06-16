# Enhancements to (in-memory) RepresentationMetadata: frames as ints, not strings.

**Created**: 2025-03-10
**Closed**: 2025-03-10
**Priority**: 3

## Description

we have a lot of `map(str, frames)` in places. `frames` in memory should be list[int] even if they are stored as strings in the JSON due to json issues.
