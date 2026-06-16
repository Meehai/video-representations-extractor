# Skip check frames skips if any format has been previously exported

**Created**: 2025-04-12
**Closed**: 2025-04-30
**Priority**: 2
**Labels**: feature

## Description

If in run 1 I export .npz only then `.repr_metadata.json` looks like this
```json
        "99": null,
        "100": {
            "run_id": "ZlBiYuP0J3",
            "duration": 0.9193933333333333
        },
```

So frame 100 is skipped at future runs. But if I want now to export `.jpg` too, it will skip it forever.

Solution: add one extra key (so we don't deep investigate in the run id which is also possible but more costly and assumes structure of dirs)
```json
        "99": null,
        "100": {
            "run_id": "ZlBiYuP0J3",
            "duration": 0.9193933333333333
            "exported_formats": ["png", "npz"]
        },
```
And we check if current run's exported formats is in the previously ones.
Note: we need to ensure skipping previous ones somehow for these frames! We can treat that as a later optimization maybe
