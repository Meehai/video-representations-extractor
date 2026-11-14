# Split the pip installation (setup.py based I guess) in 2: full & base

**Created**: 2024-10-22
**Priority**: 3
**Closed**: 2024-10-23

## Description

- full == defualt 
- base all basic deps + torch but no frameworks code (i.e. cocoutils or fvcore etc.)

Basically nothing that's in `*_impl` but only what's needed for utils/readers/basic representations.
I think cv2 is needed too sadly.
