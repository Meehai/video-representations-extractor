"""
mask2former's swin backbone used timm's DropPath; it now uses torchvision's StochasticDepth("row"). Same input, same
seed, same output, in eval (identity) and train (per-sample drop + rescale). Runs only if timm is installed.
"""
import pytest
import torch as tr
from torchvision.ops import StochasticDepth

timm = pytest.importorskip("timm")

@pytest.mark.parametrize("mode", ["eval", "train"])
def test_parity_swin_drop_path_vs_timm(mode: str):
    timm_drop = timm.layers.DropPath(0.3).train(mode == "train")
    vre_drop = StochasticDepth(0.3, "row").train(mode == "train")
    x = tr.randn(64, 49, 96, generator=tr.Generator().manual_seed(42))

    tr.manual_seed(0)
    y_timm = timm_drop(x)
    tr.manual_seed(0)
    y_vre = vre_drop(x)

    assert tr.equal(y_timm, y_vre), (y_timm - y_vre).abs().max()
    assert tr.equal(y_vre, x) == (mode == "eval")
