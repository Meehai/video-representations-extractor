# Standardize load_from_disk & return_fn everywhere (including MultiTaskReader)

**Created**: 2024-11-09
**Closed**: 2024-11-10
**Priority**: 3

## Description

- load_from_disk is what's stored on disk (and cached)
- return_fn (other name?) is what gets passed around (i.e. in TaskMapper/ML code etc.)
