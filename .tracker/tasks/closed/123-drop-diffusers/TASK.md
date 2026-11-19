# Drop diffusers: plain-torch UNet/VAE/DDIM+LCM for Marigold

**Created**: 2026-11-19
**Priority**: 2
**Closed**: 2026-11-19

**Labels**: deps

## Description

`vre_repository` must be torch/torchvision only. diffusers is there for legacy reasons: Marigold uses
`UNet2DConditionModel`, `AutoencoderKL`, `DDIMScheduler`, `LCMScheduler`, `FrozenDict`; `vre/__init__.py` imports it
to disable a progress bar (core must not touch it at all).

Same recipe as DINOv2 (#120): plain-torch port, same state_dict keys, `vae.pt` + unet shards unchanged.

- `marigold_impl/`: UNet, VAE (encoder/decoder/quant convs only), DDIM + LCM schedulers -- only the code paths the
  Marigold configs (`marigold-lcm-v1-0`, `marigold-v1-0`, `testing`) hit; assert the rest away.
- UNet returns a tensor (no `.sample`). Drop `FrozenDict`.
- `vre/__init__.py`: drop the diffusers import and the transformers-era `clean_up_tokenization_spaces` filter.
- Remove `diffusers` from `setup.py`, docs.

## Done when

- [x] code applied (!255)
- [x] bit-identical with diffusers 0.30.3 on CPU fp32: modules, schedulers
      (`test/manual/marigold/check_diffusers_parity.py`). The final version (after review fixes) was validated by CI
      e2e only; the full seeded demo1.jpg run of the parity script was not re-run on it.
- [x] nothing imports diffusers (`vre_repository` and `vre/__init__.py`)
- [x] review gate green in CI (!255)
