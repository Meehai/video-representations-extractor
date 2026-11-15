#!/usr/bin/env python3
"""
Parity of VRE's `embeddings/dinov2` against transformers' Dinov2Model, and against the visual-graph-navigation
Embedder (scripts/14-embeddings/embeddings/embedder.py), which wraps transformers with the same preprocessing/pooling.

Needs `transformers` (>= 4.45: size-based position embedding interpolation) + HF hub access. Not a CI test.
Usage: check_hf_parity.py [--embedder_dir /path/to/14-embeddings] [--variants dinov2-small dinov2-base]
Prints max |diff| per (variant, device, pool, input); 0.0 == bit-identical.
"""
import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch as tr
from transformers import AutoModel
from vre_video import VREVideo
from vre.utils import get_project_root, image_read
from vre_repository.embeddings.dinov2 import DinoV2
from vre_repository.embeddings.dinov2.dinov2_impl import Dinov2Model

parser = ArgumentParser()
parser.add_argument("--embedder_dir", type=Path,
                    default=Path("/home/mihai/code/robotics/visual-graph-navigation/scripts/14-embeddings"))
parser.add_argument("--variants", nargs="+", default=["dinov2-small", "dinov2-base", "dinov2-large"])
args = parser.parse_args()

devices = ["cpu", "cuda"] if tr.cuda.is_available() else ["cpu"]
frames = np.stack([image_read(get_project_root() / "resources/demo1.jpg")] * 2) # (2, H, W, 3)
inputs = {"frames": frames, "square518": np.ascontiguousarray(frames[:, :518, :518]),
          "portrait": np.ascontiguousarray(frames[:, :, :frames.shape[1] // 2])}

# 1. the raw model: same state_dict keys, same last_hidden_state as transformers (fp32, random input sizes)
for variant in args.variants:
    hf = AutoModel.from_pretrained(f"facebook/{variant}", attn_implementation="sdpa").eval()
    port = Dinov2Model(*{"dinov2-small": (384, 12, 6), "dinov2-base": (768, 12, 12),
                         "dinov2-large": (1024, 24, 16)}[variant]).eval()
    port.load_state_dict(tr.load(DinoV2.get_weights_paths(variant)[0], map_location="cpu"))
    assert list(hf.state_dict()) == list(port.state_dict())
    for h, w in [(518, 518), (224, 224), (378, 672), (672, 378), (126, 70)]:
        x = tr.randn(2, 3, h, w)
        with tr.no_grad():
            diff = (hf(pixel_values=x).last_hidden_state - port(x)).abs().max().item()
        print(f"[model] {variant} {h}x{w}: max|diff|={diff}")

# 2. the representation vs the Embedder (transformers inside), per device and pool
if not args.embedder_dir.exists():
    print(f"no Embedder at '{args.embedder_dir}', skipping")
    sys.exit(0)
sys.path.insert(0, str(args.embedder_dir))
from embeddings.embedder import Embedder # pylint: disable=wrong-import-position,import-error
for variant in args.variants:
    for device in devices:
        for pool in ["cls", "gem", "cat"]:
            embedder = Embedder("dinov2", checkpoint=f"facebook/{variant}", pool=pool, long_side=672, device=device)
            rep = DinoV2(variant=variant, pool=pool, long_side=672, name="dinov2", dependencies=[])
            rep.device = device
            rep.vre_setup()
            for name, x in inputs.items():
                y_vre = rep.compute(VREVideo(x, fps=1), [0, 1]).output
                y_embedder = embedder.embed(x)
                print(f"[embedder] {variant} {device} {pool} {name} {y_vre.shape}: "
                      f"max|diff|={np.abs(y_vre - y_embedder).max()}")
            rep.vre_free()
