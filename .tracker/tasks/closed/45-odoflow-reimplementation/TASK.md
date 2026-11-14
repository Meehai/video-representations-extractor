# odoflow reimplementation

**Created**: 2024-10-06
**Priority**: 3
**Closed**: 2024-10-16

## Description

the current one is almost tech debt, lots of dead code paths and stuff. We need a clear one, but it's most likely a ~1 week project so for now I'll just remove it.

```
depth[ix] <- odoflow(rgb[ix], flow[ix], flow[ix+k], sensor_fov: int, sensor_hw: tuple[int, int], angular_velocity[ix], linear_velocity[ix], intrinsics_K: 4x4)
```
