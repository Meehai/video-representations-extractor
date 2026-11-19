"""
DPT's ViT-L/16 backbone vs timm's vit_large_patch16_384: same checkpoint, two libraries, same output.
The checkpoint's `pretrained.model.*` is a timm state_dict, so it loads strict in both. Runs only if timm is installed.
Only the timm-native 384x384 input: other sizes resize the position embeddings, which is DPT's logic, not timm's.
"""
import pytest
import torch as tr
from vre_repository.depth.dpt import DepthDpt
from vre_repository.utils import fetch_weights

timm = pytest.importorskip("timm")

def test_parity_dpt_vit_vs_timm():
    ckpt = tr.load(fetch_weights(DepthDpt.get_weights_paths())[0], map_location="cpu")
    vit_ckpt = {k.removeprefix("pretrained.model."): v for k, v in ckpt.items() if k.startswith("pretrained.model.")}

    timm_vit = timm.create_model("vit_large_patch16_384", pretrained=False).eval()
    timm_vit.load_state_dict(vit_ckpt)
    dpt = DepthDpt(name="dpt", dependencies=[])
    dpt.vre_setup(load_weights=True)
    vre_vit = dpt.model.pretrained.model
    assert sorted(vre_vit.state_dict()) == sorted(timm_vit.state_dict())

    x = tr.rand(2, 3, 384, 384, generator=tr.Generator().manual_seed(42)) * 2 - 1
    with tr.no_grad():
        y_timm = timm_vit.forward_intermediates(x, indices=list(vre_vit.out_blocks), return_prefix_tokens=True,
                                                output_fmt="NLC", intermediates_only=True)
        y_vre = vre_vit(x)
    assert len(y_timm) == len(y_vre) == 4, (len(y_timm), len(y_vre))
    for (patches, cls), y in zip(y_timm, y_vre, strict=True):
        assert tr.equal(tr.cat([cls, patches], dim=1), y), (tr.cat([cls, patches], dim=1) - y).abs().max()
