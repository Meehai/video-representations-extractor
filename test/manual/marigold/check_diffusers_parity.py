#!/usr/bin/env python3
"""
Parity of VRE's `depth/marigold` plain-torch port (marigold_impl/{unet,vae,schedulers}.py) against diffusers 0.30.3.

Needs `diffusers`. Not a CI test. Run with this checkout first on the path (PYTHONPATH=.:pkg/vre-video:pkg/image_utils).
Usage: check_diffusers_parity.py [--reference_repo /tmp/vre-nodeps/diffusers-ref] [--devices cpu cuda]
                                 [--parts nets schedulers e2e]
  nets:       UNet/VAE at the 'testing' sizes (head_dim 1) and at 5x them (head_dim 8), random init in diffusers,
              state_dict copied to the port (same keys, strict). max|diff| of unet(x, t, ctx), vae encode
              (encoder -> quant_conv), decode (post_quant_conv -> decoder).
  schedulers: DDIM/LCM set_timesteps() equal, step() chains equal for seeded inputs and generators.
  e2e:        Marigold(variant).compute(resources/demo1.jpg) with the real weights, for marigold-lcm-v1-0 (LCM, 768,
              4 steps, seed 42) and marigold-v1-0 (DDIM, 512, 10 steps, seed 42). Each side runs the same code in a
              subprocess: this checkout vs --reference_repo, a VRE checkout still on diffusers
              (`git worktree add --detach /tmp/vre-nodeps/diffusers-ref dee1549`), both with this checkout's weights.
              Plus marigold.py main()'s mean/std for the LCM case.
Prints max |diff|; 0.0 == bit-identical.
"""
import os
import subprocess
import sys
import tempfile
from itertools import product
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch as tr
import diffusers
from vre.utils import get_project_root
from vre_repository.depth.marigold.marigold_impl import (AutoencoderKL, DDIMScheduler, LCMScheduler,
                                                         UNet2DConditionModel)

parser = ArgumentParser()
parser.add_argument("--reference_repo", type=Path)
parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"] if tr.cuda.is_available() else ["cpu"])
parser.add_argument("--parts", nargs="+", default=["nets", "schedulers", "e2e"])
args = parser.parse_args()
assert diffusers.__version__ == "0.30.3", diffusers.__version__

# the diffusers configs Marigold used ('testing' sizes; real: [320, 640, 1280, 1280] / [128, 256, 512, 512], 32 groups)
UNET_CFG = {
    "sample_size": 96, "in_channels": 8, "out_channels": 4, "center_input_sample": False, "flip_sin_to_cos": True,
    "freq_shift": 0, "down_block_types": ["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
    "mid_block_type": "UNetMidBlock2DCrossAttn", "up_block_types": ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3,
    "only_cross_attention": False, "block_out_channels": [8, 16, 32, 32], "layers_per_block": 2,
    "downsample_padding": 1, "mid_block_scale_factor": 1, "dropout": 0.0, "act_fn": "silu", "norm_num_groups": 8,
    "norm_eps": 1e-05, "cross_attention_dim": 1024, "transformer_layers_per_block": 1,
    "attention_head_dim": [5, 10, 20, 20], "use_linear_projection": True, "upcast_attention": False,
    "resnet_time_scale_shift": "default", "time_embedding_type": "positional", "conv_in_kernel": 3,
    "conv_out_kernel": 3, "attention_type": "default",
}
VAE_CFG = {
    "in_channels": 3, "out_channels": 3, "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4, "block_out_channels": [8, 16, 32, 32], "layers_per_block": 2,
    "act_fn": "silu", "latent_channels": 4, "norm_num_groups": 8, "sample_size": 768, "scaling_factor": 0.18215,
    "force_upcast": True, "use_quant_conv": True, "use_post_quant_conv": True, "mid_block_add_attention": True,
}
DDIM_CFG = {
    "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085, "clip_sample": False,
    "clip_sample_range": 1.0, "dynamic_thresholding_ratio": 0.995, "num_train_timesteps": 1000,
    "prediction_type": "v_prediction", "rescale_betas_zero_snr": False, "sample_max_value": 1.0,
    "set_alpha_to_one": False, "steps_offset": 1, "thresholding": False, "timestep_spacing": "leading",
    "trained_betas": None,
}
LCM_CFG = {**DDIM_CFG, "original_inference_steps": 50, "timestep_scaling": 10.0}
IMAGE = get_project_root() / "resources/demo1.jpg"
MAIN_EXPECTED = {"cpu": (0.47739768, 0.28376475), "cuda": (0.46834466, 0.32319403)} # marigold.py main(), demo1.jpg

# 1. the networks: same state_dict keys, same outputs (random weights, random inputs)
if "nets" in args.parts:
    for unet_ch, vae_ch in [([8, 16, 32, 32], [8, 16, 32, 32]), ([40, 80, 160, 160], [40, 80, 160, 160])]:
        tr.manual_seed(0)
        ref_unet = diffusers.UNet2DConditionModel(**{**UNET_CFG, "block_out_channels": unet_ch})
        ref_vae = diffusers.AutoencoderKL(**{**VAE_CFG, "block_out_channels": vae_ch})
        with tr.no_grad():
            for p in [*ref_unet.parameters(), *ref_vae.parameters()]: # off the default init (norms at 1/0, ...)
                p.add_(tr.randn_like(p) * 0.05)
        unet = UNet2DConditionModel(8, 4, unet_ch, [5, 10, 20, 20], norm_num_groups=8, cross_attention_dim=1024)
        vae = AutoencoderKL(vae_ch, norm_num_groups=8)
        assert sorted(ref_unet.state_dict()) == sorted(unet.state_dict())
        assert sorted(ref_vae.state_dict()) == sorted(vae.state_dict())
        unet.load_state_dict(ref_unet.state_dict())
        vae.load_state_dict(ref_vae.state_dict())
        for device in args.devices:
            ref_unet, ref_vae, unet, vae = [m.to(device) for m in (ref_unet, ref_vae, unet, vae)]
            for b, h, w in [(1, 16, 16), (2, 13, 21), (1, 96, 53)]:
                x, ctx = tr.randn(b, 8, h, w, device=device), tr.randn(b, 2, 1024, device=device)
                img, z = tr.randn(b, 3, h * 8, w * 8, device=device), tr.randn(b, 4, h, w, device=device)
                for t in [999, 500, 1]:
                    timestep = tr.tensor(t, device=device)
                    with tr.no_grad():
                        diff = (ref_unet(x, timestep, encoder_hidden_states=ctx).sample - unet(x, timestep, ctx))
                    print(f"[unet {unet_ch}] {device} {b}x{h}x{w} t={t}: max|diff|={diff.abs().max().item()}")
                with tr.no_grad():
                    diff_enc = (ref_vae.quant_conv(ref_vae.encoder(img)) - vae.quant_conv(vae.encoder(img))).abs()
                    diff_dec = (ref_vae.decoder(ref_vae.post_quant_conv(z)) - vae.decoder(vae.post_quant_conv(z))).abs()
                print(f"[vae {vae_ch}] {device} {b}x{h * 8}x{w * 8}: encode max|diff|={diff_enc.max().item()} "
                      f"decode max|diff|={diff_dec.max().item()}")
        del ref_unet, ref_vae, unet, vae
    tr.cuda.empty_cache() # the e2e subprocesses need ~4.4GB of VRAM

# 2. the schedulers: same timesteps, same denoising chains (random model outputs, seeded generators)
if "schedulers" in args.parts:
    for name, ref, port in [("ddim", diffusers.DDIMScheduler(**DDIM_CFG), DDIMScheduler()),
                            ("lcm", diffusers.LCMScheduler(**LCM_CFG), LCMScheduler())]:
        for device in args.devices:
            for n in [*range(1, 11), 50]:
                ref.set_timesteps(n, device=device)
                port.set_timesteps(n, device=device)
                assert ref.timesteps.dtype == port.timesteps.dtype and tr.equal(ref.timesteps, port.timesteps), n
                gen_inputs, gen_ref, gen_port = [tr.Generator(device).manual_seed(42) for _ in range(3)]
                y_ref = y_port = tr.randn(2, 4, 13, 21, device=device, generator=gen_inputs)
                diff = 0.0
                for t in ref.timesteps:
                    model_output = tr.randn(2, 4, 13, 21, device=device, generator=gen_inputs)
                    y_ref = ref.step(model_output, t, y_ref, generator=gen_ref).prev_sample
                    y_port = port.step(model_output, t, y_port, generator=gen_port)
                    diff = max(diff, (y_ref - y_port).abs().max().item())
                print(f"[scheduler] {name} {device} n={n}: timesteps equal, step chain max|diff|={diff}")

# 3. the representation with the real weights: the same code in this checkout and in the reference one (subprocesses)
E2E_CODE = """
import sys
import numpy as np
import torch as tr
from vre.utils import image_read
from vre_repository.depth.marigold import Marigold
variant, steps, res, device, image, vae_path, unet_paths, out_path = sys.argv[1:]
rep = Marigold(variant, denoising_steps=int(steps), ensemble_size=1, processing_resolution=int(res), seed=42,
               name="marigold", dependencies=[])
rep.device = device
rep.vre_setup(load_weights=False)
rep.model.vae.load_state_dict(tr.load(vae_path, map_location="cpu", mmap=True))
unet_sd = {k: v for p in unet_paths.split(",") for k, v in tr.load(p, map_location="cpu", mmap=True).items()}
rep.model.unet.load_state_dict(unet_sd)
del unet_sd
np.save(out_path, rep.compute(image_read(image)[None], [0]).output)
"""
WEIGHTS_DIR = get_project_root() / "vre_repository/depth/marigold/weights"
E2E_CASES = [("marigold-lcm-v1-0", 4, 768, [WEIGHTS_DIR / f"marigold-lcm-v1-0_unet.pt/000{i}.pt" for i in range(4)]),
             ("marigold-v1-0", 10, 512, [WEIGHTS_DIR / "marigold-v1-0_unet.pt"])]

def run_e2e(repo: Path, variant: str, steps: int, res: int, device: str, unet_paths: list[Path]) -> np.ndarray:
    """runs E2E_CODE with `repo` first on the path, returns the (1, H, W, 1) depth"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": f"{repo}:{repo}/pkg/vre-video:{repo}/pkg/image_utils"}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-c", E2E_CODE, variant, str(steps), str(res), device, str(IMAGE),
                        str(WEIGHTS_DIR / "vae.pt"), ",".join(map(str, unet_paths)), f"{tmp}/y.npy"],
                       env=env, cwd=repo, check=True)
        return np.load(f"{tmp}/y.npy")

if "e2e" in args.parts:
    for (variant, steps, res, unet_paths), device in product(E2E_CASES, args.devices):
        y = run_e2e(get_project_root(), variant, steps, res, device, unet_paths)
        y_ref = None if args.reference_repo is None else run_e2e(args.reference_repo.absolute(), variant, steps,
                                                                    res, device, unet_paths)
        diff = "n/a (no --reference_repo)" if y_ref is None else np.abs(y_ref - y).max()
        mean_std = (float(y.mean()), float(y.std()))
        main_check = ""
        if variant == "marigold-lcm-v1-0":
            ok = np.allclose(MAIN_EXPECTED[device], mean_std)
            main_check = f" main() expects {MAIN_EXPECTED[device]}: {'ok' if ok else 'MISMATCH'}"
        print(f"[e2e] {variant} {res} {steps} steps {device} {y.shape}: max|diff|={diff} mean,std={mean_std}"
              f"{main_check}")
