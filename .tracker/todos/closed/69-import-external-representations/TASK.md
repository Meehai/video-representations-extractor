# Load/Import externally built representations using the vre cli tool

**Created**: 2024-11-10
**Closed**: 2024-11-12
**Priority**: 1
**Labels**: priority

## Description

Something like
```vre /path/to/video.mp4 --config_path /path/to/cfg.yaml -I /path/to/external_repr_code.py```
and this code must have an `VRE_EXPORTRED_REPRESENTATIONS: dict[str, Representation]` dict defined that is loaded dynamically.
