# collage fails (2 external repr?)

**Created**: 2025-01-21
**Closed**: 2025-01-21
**Priority**: 3

## Description

```
vre_collage ../data/train_set_experts/ -o collages/train_set_experts --config_path cfg.yaml -I ../scripts/semantic_mapper/semantic_mapper.py:get_new_semantic_mapped_tasks ../scripts/dronescapes_viewer/dronescapes_representations.py:get_gt_tasks --video --fps 30 --overwrite --output_resolution 720 1280 --n_workers 8
```
