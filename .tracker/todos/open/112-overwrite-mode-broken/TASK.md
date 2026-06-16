# --output_dir_exists_mode=overwrite doesn't work anymore

**Created**: 2025-04-12
**Priority**: 3

## Description

frames are skipped always due to metadata

also, if I patch it:

```python

        relevant_frames = runtime_args.frames
        if output_dir_exists_mode == "skip_computed":
            relevant_frames = [f for f in runtime_args.frames if f not in repr_metadata.frames_computed()]
            logger.debug(f"Out of {len(runtime_args.frames)} total frames, "
                         f"{len(runtime_args.frames) - len(relevant_frames)} are precomputed and will be skipped.")
```

it fails later
