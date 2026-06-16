# Skip frames check is wrong: no 'reality check'

**Created**: 2025-04-12
**Priority**: 1
**Labels**: bug

## Description

If  `.repr_metadata.json` looks like this:
```json
        "99": null,
        "100": {
            "run_id": "ZlBiYuP0J3",
            "duration": 0.9193933333333333
        },
```

It doesn't check at all if the frame exists as it assumes that it's not corrupted.

To decide: is such a check worth doing (maybe some flag?). Imho yes, a disk check should be faster than a bug later on.
