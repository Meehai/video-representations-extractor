# Integration test: task_types across two dataset objects (train/val) w/ TaskMapping

**Created**: 2025-01-28
**Priority**: 3

## Description

bug is fixed here: https://gitlab.com/video-representations-extractor/video-representations-extractor/-/merge_requests/186

but a test would be nice for regressions. See here: https://gitlab.com/video-representations-extractor/neo-transformers/-/blob/master/train.py?ref_type=heads#L40

one of the task_types must be TaskMapped (.dependencies updated in reader).
