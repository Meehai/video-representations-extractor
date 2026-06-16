# wrong final metadata stdout string

**Created**: 2025-03-03
**Closed**: 2025-03-10
**Priority**: 3

## Description

IF we export only a repr in a vre dir with pre-existing reprs, it shows the wrong one at the end:

```
[VRE] semantic_mask2former_swin_mapillary_converted bs=1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.57it/s]
ix   |depth_marigold      |semantic_mask..528_1|semantic_mask..528_0|semantic_mask..163_0|rgb                 |normals_svd(d..gold)|semantic_mask..erted|semantic_mask..erted|semantic_mask..erted|buildings           |sky-and-water       |transportation      |containing          |vegetation          |buildings(nearby)   |safe-landing-no-sseg|safe-landing-..ntics|semantic_output     
131  |0.171               
308  |0.165               
356  |0.167               
422  |0.164               
559  |0.228               

Total:
{'depth_marigold': 911.171}

(ngc) mihai_cristian.pirvu[collage_comparison]$ CUDA_VISIBLE_DEVICES=0 VRE_DEVICE=cuda vre bucharest_youtube_540p.mp4 --frames 0..967 --config_path /scratch/sdc/datasets/dronescapes-2024/scripts/collage_comparison/cfg.yaml -o data_bucharest_youtube_540p.mp4 --representations semantic_mask2former_swin_mapillary_converted -I /export/home/proiecte/aux/mihai_cristian.pirvu/code/neo-transformers/readers/semantic_mapper.py:get_new_semantic_mapped_tasks --output_dir_exists_mode skip_computed
```
