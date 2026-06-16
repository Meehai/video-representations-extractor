# export_npy as float16 argument

**Created**: 2024-05-09
**Closed**: 2024-10-24
**Priority**: 3

## Description

and check that if it is float, it's float32, otherwise float16. We don't touch uint8.

Maybe we need a `representation.cast("float16")` :/ ?
