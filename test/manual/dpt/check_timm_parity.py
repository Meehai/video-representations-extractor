#!/usr/bin/env python3
"""
Parity of VRE's plain-torch DPT (`depth/dpt/dpt_impl`) and mask2former swin backbone against their timm-based
versions at git ref `--ref` (default dee1549, the last commit that imports timm), extracted with `git show`.

Needs `timm` (1.0.9) installed + the real DPT weights (git-lfs). Not a CI test.
Usage: check_timm_parity.py [--ref dee1549]  (wrap in `flock` when the GPU is shared)
Prints max |diff| per (model, case, device, dtype); 0.0 == bit-identical.
"""
import gc
import json
import sys
import subprocess
import importlib.util
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import torch as tr
from vre.utils import get_project_root
from vre_repository.depth.dpt import DepthDpt
from vre_repository.depth.dpt.dpt_impl import DPTDepthModel
from vre_repository.semantic_segmentation.mask2former.mask2former_impl import CfgNode
from vre_repository.semantic_segmentation.mask2former.mask2former_impl.modeling.backbone import swin

def import_from_ref(ref: str, rel_paths: list[str], name: str, out_dir: Path) -> ModuleType:
    """the files `rel_paths` at git `ref` written to `out_dir`, the first one imported as module `name`. If it is an
    __init__.py, the rest are its package's submodules (relative imports resolve to them)."""
    for rel_path in rel_paths:
        out_path = out_dir / name / Path(rel_path).name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(subprocess.check_output(["git", "-C", str(get_project_root()), "show", f"{ref}:{rel_path}"],
                                                    text=True))
    path = out_dir / name / Path(rel_paths[0]).name
    is_pkg = path.name == "__init__.py"
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)]
                                                  if is_pkg else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

parser = ArgumentParser()
parser.add_argument("--ref", default="dee1549")
args = parser.parse_args()
tmp_dir = Path(TemporaryDirectory().name)
tmp_dir.mkdir(parents=True)

devices = ["cpu", "cuda"] if tr.cuda.is_available() else ["cpu"]
runs = [("cpu", "fp32")] + ([("cuda", "fp32"), ("cuda", "fp16")] if "cuda" in devices else [])
gen = tr.Generator().manual_seed(42)
dpt_cases = {"384x384": tr.rand(1, 3, 384, 384, generator=gen) * 2 - 1,
             "384x672": tr.rand(1, 3, 384, 672, generator=gen) * 2 - 1,
             "672x384": tr.rand(1, 3, 672, 384, generator=gen) * 2 - 1,
             "192x384": tr.rand(1, 3, 192, 384, generator=gen) * 2 - 1,
             "b2_384x384": tr.rand(2, 3, 384, 384, generator=gen) * 2 - 1}
ckpt = tr.load(DepthDpt.get_weights_paths()[0], map_location="cpu")

def run_model(model: tr.nn.Module, cases: dict[str, tr.Tensor]) -> dict[tuple[str, str, str], tr.Tensor]:
    """the model's output per (case, device, dtype), on cpu"""
    res = {}
    for run_device, run_dtype in runs:
        model = model.to(run_device)
        for name, x in cases.items():
            with tr.no_grad(), tr.autocast("cuda", dtype=tr.float16, enabled=run_dtype == "fp16"):
                res[(name, run_device, run_dtype)] = model(x.to(run_device)).float().cpu()
        model = model.to("cpu")
    return res

# 1. DPT: the timm model (one at a time: ViT-L is ~1.3GB), then the port. Same checkpoint, strict load in both.
DPT_DIR = "vre_repository/depth/dpt/dpt_impl"
DPT_FILES = ["__init__", "base_model", "blocks", "dpt_depth", "utils", "vit"]
dpt_impl_timm = import_from_ref(args.ref, [f"{DPT_DIR}/{f}.py" for f in DPT_FILES], "dpt_impl_timm", tmp_dir)
old = dpt_impl_timm.DPTDepthModel(backbone="vitl16_384", non_negative=True).eval()
old.load_state_dict(ckpt)
old_shapes = {k: v.shape for k, v in old.state_dict().items()}
y_old = run_model(old, dpt_cases)
del old
gc.collect()
tr.cuda.empty_cache()

new = DPTDepthModel().eval()
new.load_state_dict(ckpt)
new_shapes = {k: v.shape for k, v in new.state_dict().items()}
assert old_shapes == new_shapes, (set(old_shapes) ^ set(new_shapes))
assert set(new_shapes) == set(ckpt), set(new_shapes) ^ set(ckpt)
print(f"[dpt] state_dict: {len(new_shapes)} keys, same names + shapes as timm and as the checkpoint")
y_new = run_model(new, dpt_cases)
for key, y in y_new.items():
    print(f"[dpt] {key[0]:>10} {key[1]} {key[2]} {tuple(y.shape)} mean={y.mean().item():.4f}: "
          f"max|diff|={(y - y_old[key]).abs().max().item()}")
del new
gc.collect()
tr.cuda.empty_cache()

# 2. swin (mask2former 47429163_0 config): the timm module's random init loaded into the new one, same outputs.
# eval (DropPath == identity) and train with the same seed (DropPath draws: timm vs torchvision StochasticDepth).
SWIN_REL_PATH = "vre_repository/semantic_segmentation/mask2former/mask2former_impl/modeling/backbone/swin.py"
swin_timm = import_from_ref(args.ref, [SWIN_REL_PATH], f"{swin.__name__}_timm", tmp_dir) # same package as the new one
cfg_path = Path(swin.__file__).parents[2] / "resources/47429163_0_cfg.json"
cfg = CfgNode(json.load(open(cfg_path, "r", encoding="utf-8")))
tr.manual_seed(0)
old_swin = swin_timm.D2SwinTransformer(cfg, None)
tr.manual_seed(0)
new_swin = swin.D2SwinTransformer(cfg, None)
init_diff = max((a - b).abs().max().item() for a, b in zip(old_swin.state_dict().values(),
                                                              new_swin.state_dict().values(), strict=True))
print(f"[swin] same-seed init max|diff|={init_diff} (trunc_normal_: timm inverse-cdf vs torch's sampler)")
assert list(old_swin.state_dict()) == list(new_swin.state_dict())
new_swin.load_state_dict(old_swin.state_dict())
x_swin = tr.randn(2, 3, 384, 512, generator=gen)
for device, dtype in runs:
    for mode in ["eval", "train"]:
        outs = []
        for swin_model in [old_swin, new_swin]:
            swin_model.to(device).train(mode == "train") # swin's train() returns None: no chaining
            tr.manual_seed(1)
            with tr.no_grad(), tr.autocast("cuda", dtype=tr.float16, enabled=dtype == "fp16"):
                outs.append({k: v.float().cpu() for k, v in swin_model(x_swin.to(device)).items()})
        diff = max((outs[0][k] - outs[1][k]).abs().max().item() for k in outs[0])
        print(f"[swin] {mode:>5} {device} {dtype} {sorted(outs[0])}: max|diff|={diff}")
