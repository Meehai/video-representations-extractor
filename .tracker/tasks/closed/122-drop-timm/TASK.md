# Drop timm: plain-torch ViT-L/16 for DPT, torch helpers in swin

**Created**: 2026-11-19
**Priority**: 2
**Closed**: 2026-11-19

**Labels**: deps

## Description

`vre_repository` must be torch/torchvision only. timm is there for legacy reasons:

- `depth/dpt/dpt_impl/vit.py`: `timm.create_model("vit_large_patch16_384")`. DepthDpt only builds `vitl16_384` + `project` readout.
- `mask2former/.../backbone/swin.py`: `DropPath`, `to_2tuple`, `trunc_normal_`.

Same recipe as DINOv2 (#120): plain-torch port, same state_dict keys, weights file unchanged.

- Minimal ViT-L/16 (cls token, pos_embed resized per input, 24 blocks). Return the 4 hooked block outputs from forward
  (no module-global `activations` dict + hooks).
- Drop dead backbones (`vitb_rn50_384`, `vitb16_384`, deit, `resnext101_wsl`).
- swin: `torch.nn.init.trunc_normal_`, `torchvision.ops.StochasticDepth` / tiny DropPath, `(x, x)`.
- Remove `timm` from `setup.py`, docs.

## Done when

- [x] code applied (!254)
- [x] bit-identical with timm 1.0.9 (`test/manual/dpt/check_timm_parity.py`): DPT + swin max|diff| 0.0 on cpu fp32,
      cuda fp32, cuda fp16; 368 keys, `depth_dpt_midas.pth` unchanged
- [x] nothing imports timm (grep)
- [x] review gate green in CI (!254)
