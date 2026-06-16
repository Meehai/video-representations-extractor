# vre_setup vs __init_() setup + one in/out test for each representation

**Created**: 2023-11-06
**Closed**: 2023-11-06
**Priority**: 3

## Description

- vre_setup -> loading weights from vre weights repository
- _setup() is just __init__() -> model intantiation

- each representation has a I/O unit test that validates that we can run the repr w/o doing vre_setup() on it
  - allows easy refactoring, like getting cv2 out (#16)
