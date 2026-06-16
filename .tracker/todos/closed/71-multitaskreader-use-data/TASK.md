# MultiTaskReader refactor: use .data from representations instead of the low level load_from_disk/from_disk_fmt ops

**Created**: 2024-11-10
**Closed**: 2024-11-14
**Priority**: 3

## Description

we need a way to abstract that part away and unify a bit the reader and vre code on the loading part
