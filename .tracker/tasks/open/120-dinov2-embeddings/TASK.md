# Representation: DINOv2 embeddings (torch-only, no transformers)

**Created**: 2026-11-14
**Priority**: 2
**Closed**: n/a

**Labels**: representation

## Description

Add DINOv2 as `embeddings/dinov2`: one global L2-normalized vector per frame (pool: cls | gem | cat), the same
embedder visual-graph-navigation uses (`scripts/14-embeddings/embeddings/embedder.py`), so VRE outputs can replace it.

- `vre_repository/embeddings/` new category: `EmbeddingRepresentation` owns what every global vector shares
  (resize keeps the vector, make_images draws it as a grid at the frame size). `descriptor/` stays keypoints+descriptors.
- `dinov2/dinov2_impl.py`: plain-torch port of transformers' `Dinov2Model`, same state_dict keys.
- Weights: `weights/dinov2-{small,base,large}.pt` = HF `facebook/dinov2-*` state_dicts (git-lfs, `git add -f`).
- Giant is out of scope (SwiGLU MLP + 4.5GB sharded weights): add when someone needs it.

## Done when

- [ ] code applied (developer), registered in `get_vre_repository()`
- [ ] weights committed + pushed to master (CI fetches them from master's raw URL)
- [x] bit-identical with transformers 4.45 on CPU fp32 and cuda fp16 autocast (`test/manual/dinov2/check_hf_parity.py`)
- [x] unit + integration tests, `vre_batched` e2e cfg
- [ ] review gate green in CI

Related: 78 (example: extract visual features).
