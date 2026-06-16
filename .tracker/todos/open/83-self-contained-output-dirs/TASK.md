# Make the output_dirs self contained for vre_collage and vre_reader

**Created**: 2024-11-18
**Priority**: 2
**Labels**: feature

## Description

from the code:
```
    # TODO: can we make the VRE-dir self contained so we don't need the original cfg again?
    # (representation.save_cfg() might be useful here) -- especially for external representations

```

Right now both of them require to use the `cfg.yaml` from `vre`. This can be stored in a hidden file in each `out_dir/repr_name` dir perhaps.
