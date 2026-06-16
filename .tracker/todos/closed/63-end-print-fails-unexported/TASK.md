# the print at the end of vre fails for representations that are not exported

**Created**: 2024-10-29
**Closed**: 2024-11-09
**Priority**: 3

## Description

`vre_run_stats = pd.DataFrame(vre_metadata["run_stats"], index=range(*vre_metadata["runtime_args"]["frames"]))`

fails with

```ValueError: Length of values (0) does not match length of index (1000)```
