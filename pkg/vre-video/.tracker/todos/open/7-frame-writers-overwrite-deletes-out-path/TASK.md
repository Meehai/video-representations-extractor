# Frame writers: overwrite=True deletes out_path, then crashes

**Created**: 2026-11-19
**Priority**: 2

## Issue

`PILFrameWriter.write` and `NumpyFrameWriter.write` with `overwrite=True` on a non-empty `out_path` call
`shutil.rmtree(out_path)` and never recreate the dir. The first frame write then fails. The `overwrite` path
is broken whenever it actually does something. `FFmpegFrameWriter` is fine (single file, parent dir mkdir'd).

## Reproduction

```
mkdir -p /tmp/ovw/x && touch /tmp/ovw/x/a
python -c 'import numpy as np; from vre_video import VREVideo
VREVideo(np.zeros((2, 8, 8, 3), np.uint8), fps=1).write("/tmp/ovw/x", overwrite=True, writer="pillow", fmt="png")'
# FileNotFoundError: [Errno 2] No such file or directory: '/tmp/ovw/x/0.png'
```

## Fix

Clear first, then mkdir: e.g. `if overwrite: shutil.rmtree(out_path, ignore_errors=True)` before
`out_path.mkdir(...)`, then keep the "non-empty -> ValueError" check. Add unit tests for both writers:
non-empty dir + `overwrite=True` writes all frames; non-empty dir + `overwrite=False` raises `ValueError`.

Not hit by VRE (it never calls `VREVideo.write`).
