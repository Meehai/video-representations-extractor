"""
DINOv2 with the real weights. The expected values come from transformers' facebook/dinov2-* Dinov2Model (same weights,
same preprocessing: resize to input_size, ImageNet norm) with each pool on top, on CPU: VRE's port must reproduce them.
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
import numpy as np
from PIL import Image
import pytest
from vre_video import VREVideo
from vre import VRE
from vre.utils import get_project_root
from vre_repository.embeddings.dinov2 import DinoV2
from vre_repository.embeddings.dinov2 import dinov2 as dinov2_module

EXPECTED = { # (variant, pool): (demo1[:4], demo2[:4], demo1 . demo2, demo1 . flipped(demo1)). gem: p=3
    ("dinov2-small", "cls"): ([-0.102709, 0.0147829, 0.0564079, 0.0298884],
                              [0.0318757, 0.0230953, -0.00246576, 0.0586057], -0.045142535, 0.72496313),
    ("dinov2-small", "gem"): ([0.0346559, 0.104428, 0.0464849, 0.0392595],
                              [0.0686632, 0.121589, 0.02181, 0.0551822], 0.83756423, 0.97898793),
    ("dinov2-small", "mean"): ([0.0149871, -0.00789142, 0.0436045, 0.016653],
                               [0.0633937, -0.0121188, -0.0746009, 0.031842], -0.014604863, 0.90289438),
    ("dinov2-base", "cls"): ([0.11636, 0.0448041, 0.0189191, -0.0339197],
                             [0.00101438, 0.0198928, 0.00119662, 0.00460864], -0.01913213, 0.80284947),
    ("dinov2-base", "gem"): ([0.0355052, 0.0342515, 0.0199947, 0.013048],
                             [0.0253551, 0.0501856, 0.0393231, 0.0222977], 0.89182383, 0.996095),
    ("dinov2-base", "mean"): ([0.022196, 0.00404745, -0.0570168, -0.0461193],
                              [-0.00292246, 0.0211059, 0.0297384, -0.0148031], 0.11731645, 0.97709227),
}

# CI runs dinov2-small only (85MB); base (331MB) is the same code at a bigger size
@pytest.mark.parametrize("pool", ["cls", "gem", "mean"])
@pytest.mark.parametrize("variant", ["dinov2-small", pytest.param("dinov2-base", marks=pytest.mark.skipif(
    "CI" in os.environ, reason="331MB checkpoint: too slow for CI, run locally"))])
def test_i_dinov2_matches_transformers(variant: str, pool: str):
    demo1 = np.array(Image.open(get_project_root() / "resources/demo1.jpg").convert("RGB"))
    demo2 = np.array(Image.open(get_project_root() / "resources/demo2.jpg").convert("RGB"))
    dinov2_demo1 = DinoV2(variant, pool=pool, input_size=(364, 672), name="dinov2", dependencies=[]) # 349x640
    dinov2_demo2 = DinoV2(variant, pool=pool, input_size=(504, 672), name="dinov2", dependencies=[]) # 960x1280
    dinov2_demo1.vre_setup() # fetches the weights if needed
    dinov2_demo2.vre_setup()
    y1 = dinov2_demo1.compute(VREVideo(demo1[None], fps=1), [0]).output[0]
    y2 = dinov2_demo2.compute(VREVideo(demo2[None], fps=1), [0]).output[0]
    y1_flip = dinov2_demo1.compute(VREVideo(demo1[None, :, ::-1].copy(), fps=1), [0]).output[0]

    exp_y1, exp_y2, exp_y1_y2, exp_y1_flip = EXPECTED[(variant, pool)]
    assert np.allclose(y1[0:4], exp_y1, atol=1e-4), y1[0:4]
    assert np.allclose(y2[0:4], exp_y2, atol=1e-4), y2[0:4]
    assert np.isclose(y1 @ y2, exp_y1_y2, atol=1e-4), y1 @ y2
    assert np.isclose(y1 @ y1_flip, exp_y1_flip, atol=1e-4), y1 @ y1_flip
    assert y1 @ y1_flip > y1 @ y2 # same place (mirrored) is closer than another place

def test_i_dinov2_vre_run():
    video = VREVideo(np.random.randint(0, 255, size=(3, 60, 80, 3), dtype=np.uint8), fps=30)
    dinov2_repr = DinoV2("dinov2-small", pool="mean", input_size=(42, 56), name="dinov2", dependencies=[])
    dinov2_repr.set_compute_params(batch_size=2)
    vre = VRE(video=video, representations=[dinov2_repr])
    vre.set_io_parameters(binary_format="npz", image_format="png", compress=True, output_size=(30, 40))
    vre.run(output_dir := Path(TemporaryDirectory().name))

    y = np.stack([dinov2_repr.load_from_disk(output_dir / f"dinov2/npz/{i}.npz") for i in range(3)])
    assert y.shape == (3, 384) and y.dtype == np.float32 # output_size does not touch the vectors
    assert np.allclose(np.linalg.norm(y, axis=1), 1, atol=1e-5)
    dinov2_repr.vre_setup() # vre.run() frees the model when done
    assert np.allclose(dinov2_repr.compute(video, [0, 1, 2]).output, y, atol=1e-5) # what was stored is what we compute
    assert np.array(Image.open(output_dir / "dinov2/png/0.png")).shape == (30, 40, 3) # images do follow output_size

def test_i_dinov2_weights_from_huggingface(monkeypatch: pytest.MonkeyPatch):
    def gitlab_404(paths: list[Path]):
        raise HTTPError(url=str(paths[0]), code=404, msg="Not Found", hdrs=None, fp=None)
    monkeypatch.setattr(dinov2_module, "fetch_weights", gitlab_404) # stub VRE's gitlab only: huggingface is real
    monkeypatch.setattr(DinoV2, "get_weights_paths", lambda variant: [Path(TemporaryDirectory().name) / "w.pt"])
    demo1 = np.array(Image.open(get_project_root() / "resources/demo1.jpg").convert("RGB"))
    dinov2_repr = DinoV2("dinov2-small", pool="cls", input_size=(364, 672), name="dinov2", dependencies=[])
    dinov2_repr.vre_setup() # downloads facebook/dinov2-small's pytorch_model.bin and loads it (strict)
    y1 = dinov2_repr.compute(VREVideo(demo1[None], fps=1), [0]).output[0]
    assert np.allclose(y1[0:4], EXPECTED[("dinov2-small", "cls")][0], atol=1e-4), y1[0:4]

def test_i_dinov2_tokens_vre_run():
    video = VREVideo(np.random.randint(0, 255, size=(3, 60, 80, 3), dtype=np.uint8), fps=30)
    dinov2_repr = DinoV2("dinov2-small", pool="none", input_size=(42, 56), name="dinov2", dependencies=[])
    dinov2_repr.set_compute_params(batch_size=2)
    vre = VRE(video=video, representations=[dinov2_repr])
    vre.set_io_parameters(binary_format="npz", image_format="png", compress=True, output_size=(30, 40))
    vre.run(output_dir := Path(TemporaryDirectory().name))

    y = np.stack([dinov2_repr.load_from_disk(output_dir / f"dinov2/npz/{i}.npz") for i in range(3)])
    assert y.shape == (3, 3, 4, 384) and y.dtype == np.float32 # (42/14, 56/14) grid; output_size does not touch it
    dinov2_repr.vre_setup() # vre.run() frees the model when done
    y_computed = dinov2_repr.compute(video, [0, 1, 2]).output # bs=3 here, bs=2 in the run: float noise on raw tokens
    assert np.allclose(y_computed, y, atol=1e-3), np.abs(y_computed - y).max() # what was stored is what we compute
    assert np.array(Image.open(output_dir / "dinov2/png/0.png")).shape == (30, 40, 3) # images do follow output_size
