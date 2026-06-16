# add run id to representation metadata on each frame & get rid of 1<<31

**Created**: 2025-03-11
**Closed**: 2025-03-12
**Priority**: 3

## Description

Now we have the following format:
```
frames:
  i: null or 1<<31 or positive float
```

- If 1<<31 -> it failed in a previous run
- If null -> it wasn't computed yet
- If positive float -> it was computed at some point


We can have the following format:
```
frames:
  i: null OR {run_id: ID, duration: positive float OR null}
```

- If null -> it wasn't computed yet
- If not null and duration is null -> it failed on the run ID
- If not null and duration is not null -> it was computed on the run ID
- cannot be: not null and run ID is not null -> error
