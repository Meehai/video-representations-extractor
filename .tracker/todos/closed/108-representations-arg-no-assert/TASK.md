# --representations CLI arg doesn't assert if the representations are not in the loaded ones

**Created**: 2025-04-06
**Closed**: 2025-04-12
**Priority**: 3

## Description

`vre video.mp4 --representations rgb some_other` seems to not fail if `some_other` doesn't exist.
