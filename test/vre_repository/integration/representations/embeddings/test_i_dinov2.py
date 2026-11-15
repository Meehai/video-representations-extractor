"""
DINOv2 with the real weights. The expected values come from transformers' facebook/dinov2-* Dinov2Model with the same
preprocessing (resize long side to 672, ImageNet norm) and 'cat' pooling, on CPU: VRE's port must reproduce them.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
import numpy as np
import pytest
from vre_video import VREVideo
from vre import VRE
from vre.utils import get_project_root, image_read
from vre_repository.embeddings.dinov2 import DinoV2
from vre_repository.embeddings.dinov2 import dinov2 as dinov2_module

EXPECTED = { # variant: (demo1[:4], demo2[:4], demo1 . demo2, demo1 . flipped(demo1))
    "dinov2-small": ([-0.0726261, 0.0104531, 0.0398863, 0.0211342], [0.0225395, 0.0163309, -0.0017436, 0.0414406],
                     0.39621085, 0.8519756),
    "dinov2-base": ([0.082279, 0.0316813, 0.0133779, -0.0239848], [0.0007172, 0.0140663, 0.0008461, 0.0032588],
                    0.43634585, 0.89947236),
}

@pytest.mark.parametrize("variant", ["dinov2-small", "dinov2-base"])
def test_i_dinov2_matches_transformers(variant: str):
    demo1 = image_read(get_project_root() / "resources/demo1.jpg")
    demo2 = image_read(get_project_root() / "resources/demo2.jpg")
    dinov2_repr = DinoV2(variant, pool="cat", long_side=672, name="dinov2", dependencies=[])
    dinov2_repr.vre_setup() # fetches the weights if needed
    y1 = dinov2_repr.compute(VREVideo(demo1[None], fps=1), [0]).output[0]
    y2 = dinov2_repr.compute(VREVideo(demo2[None], fps=1), [0]).output[0]
    y1_flip = dinov2_repr.compute(VREVideo(demo1[None, :, ::-1].copy(), fps=1), [0]).output[0]

    exp_y1, exp_y2, exp_y1_y2, exp_y1_flip = EXPECTED[variant]
    assert np.allclose(y1[0:4], exp_y1, atol=1e-4), y1[0:4]
    assert np.allclose(y2[0:4], exp_y2, atol=1e-4), y2[0:4]
    assert np.isclose(y1 @ y2, exp_y1_y2, atol=1e-4), y1 @ y2
    assert np.isclose(y1 @ y1_flip, exp_y1_flip, atol=1e-4), y1 @ y1_flip
    assert y1 @ y1_flip > y1 @ y2 # same place (mirrored) is closer than another place

def test_i_dinov2_vre_run():
    video = VREVideo(np.random.randint(0, 255, size=(3, 60, 80, 3), dtype=np.uint8), fps=30)
    dinov2_repr = DinoV2("dinov2-small", pool="cat", long_side=56, name="dinov2", dependencies=[])
    dinov2_repr.set_compute_params(batch_size=2)
    vre = VRE(video=video, representations=[dinov2_repr])
    vre.set_io_parameters(binary_format="npz", image_format="png", compress=True, output_size=(30, 40))
    vre.run(output_dir := Path(TemporaryDirectory().name))

    y = np.stack([dinov2_repr.load_from_disk(output_dir / f"dinov2/npz/{i}.npz") for i in range(3)])
    assert y.shape == (3, 768) and y.dtype == np.float32 # output_size does not touch the vectors
    assert np.allclose(np.linalg.norm(y, axis=1), 1, atol=1e-5)
    dinov2_repr.vre_setup() # vre.run() frees the model when done
    assert np.allclose(dinov2_repr.compute(video, [0, 1, 2]).output, y, atol=1e-5) # what was stored is what we compute
    assert image_read(output_dir / "dinov2/png/0.png").shape == (30, 40, 3) # images do follow output_size

def test_i_dinov2_weights_from_huggingface(monkeypatch: pytest.MonkeyPatch):
    def gitlab_404(paths: list[Path]):
        raise HTTPError(url=str(paths[0]), code=404, msg="Not Found", hdrs=None, fp=None)
    monkeypatch.setattr(dinov2_module, "fetch_weights", gitlab_404) # stub VRE's gitlab only: huggingface is real
    monkeypatch.setattr(DinoV2, "get_weights_paths", lambda variant: [Path(TemporaryDirectory().name) / "w.pt"])
    demo1 = image_read(get_project_root() / "resources/demo1.jpg")
    dinov2_repr = DinoV2("dinov2-small", pool="cat", long_side=672, name="dinov2", dependencies=[])
    dinov2_repr.vre_setup() # downloads facebook/dinov2-small's pytorch_model.bin and loads it (strict)
    y1 = dinov2_repr.compute(VREVideo(demo1[None], fps=1), [0]).output[0]
    assert np.allclose(y1[0:4], EXPECTED["dinov2-small"][0], atol=1e-4), y1[0:4]
