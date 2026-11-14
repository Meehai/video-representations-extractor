# Fix pdb keystrokes issue

**Created**: 2025-02-02
**Priority**: 1
**Closed**: 2025-02-02

**Labels**: bug

## Description

when using `breakpoint()` VRE seems to somehow make the keystrokes disappear. This happens after `FFmpegVideo.__getitem__`, most likely something to do with subprocessing.
