"""
Marigold's UNet, VAE and DDIM/LCM schedulers vs diffusers': same checkpoint, two libraries, same output.
VRE's side is the real representation (Marigold.vre_setup + MarigoldPipeline.single_infer); diffusers' side is its
modules with the same weights driven by the Marigold denoising loop. Runs only if diffusers is installed.
"""
import pytest
import torch as tr
from torch.nn import functional as F
from vre.utils import get_project_root, image_read
from vre_repository.depth.marigold import Marigold
from vre_repository.depth.marigold.marigold_impl import DDIMScheduler, LCMScheduler
from vre_repository.utils import fetch_weights, vre_load_weights

diffusers = pytest.importorskip("diffusers")

UNET_CFG = { # marigold-lcm-v1-0's unet/config.json (Stable Diffusion 2 UNet, 8 input channels: rgb + depth latents)
    "sample_size": 96, "in_channels": 8, "out_channels": 4, "center_input_sample": False, "flip_sin_to_cos": True,
    "freq_shift": 0, "down_block_types": ["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
    "mid_block_type": "UNetMidBlock2DCrossAttn", "up_block_types": ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3,
    "only_cross_attention": False, "block_out_channels": [320, 640, 1280, 1280], "layers_per_block": 2,
    "downsample_padding": 1, "mid_block_scale_factor": 1, "dropout": 0.0, "act_fn": "silu", "norm_num_groups": 32,
    "norm_eps": 1e-05, "cross_attention_dim": 1024, "transformer_layers_per_block": 1,
    "attention_head_dim": [5, 10, 20, 20], "use_linear_projection": True, "upcast_attention": False,
    "resnet_time_scale_shift": "default", "time_embedding_type": "positional", "conv_in_kernel": 3,
    "conv_out_kernel": 3, "attention_type": "default",
}
VAE_CFG = { # marigold-lcm-v1-0's vae/config.json
    "in_channels": 3, "out_channels": 3, "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4, "block_out_channels": [128, 256, 512, 512], "layers_per_block": 2,
    "act_fn": "silu", "latent_channels": 4, "norm_num_groups": 32, "sample_size": 768, "scaling_factor": 0.18215,
    "force_upcast": True, "use_quant_conv": True, "use_post_quant_conv": True, "mid_block_add_attention": True,
}
DDIM_CFG = { # marigold-lcm-v1-0's scheduler/scheduler_config.json, without the LCM-only keys
    "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085, "clip_sample": False,
    "clip_sample_range": 1.0, "dynamic_thresholding_ratio": 0.995, "num_train_timesteps": 1000,
    "prediction_type": "v_prediction", "rescale_betas_zero_snr": False, "sample_max_value": 1.0,
    "set_alpha_to_one": False, "steps_offset": 1, "thresholding": False, "timestep_spacing": "leading",
    "trained_betas": None,
}
LCM_CFG = {**DDIM_CFG, "original_inference_steps": 50, "timestep_scaling": 10.0}

@pytest.mark.parametrize("name", ["ddim", "lcm"])
def test_parity_marigold_schedulers_vs_diffusers(name: str):
    lib_scheduler = diffusers.DDIMScheduler(**DDIM_CFG) if name == "ddim" else diffusers.LCMScheduler(**LCM_CFG)
    vre_scheduler = DDIMScheduler() if name == "ddim" else LCMScheduler()
    for n in [1, 2, 4, 10, 50]:
        lib_scheduler.set_timesteps(n)
        vre_scheduler.set_timesteps(n)
        assert tr.equal(lib_scheduler.timesteps, vre_scheduler.timesteps), (n, lib_scheduler.timesteps)

        gen_inputs, gen_lib, gen_vre = [tr.Generator().manual_seed(42) for _ in range(3)]
        y_lib = y_vre = tr.randn(2, 4, 13, 21, generator=gen_inputs)
        for t in lib_scheduler.timesteps:
            model_output = tr.randn(2, 4, 13, 21, generator=gen_inputs)
            y_lib = lib_scheduler.step(model_output, t, y_lib, generator=gen_lib).prev_sample
            y_vre = vre_scheduler.step(model_output, t, y_vre, generator=gen_vre)
            assert tr.equal(y_lib, y_vre), (n, t, (y_lib - y_vre).abs().max())

def test_parity_marigold_lcm_vs_diffusers():
    vae_path, unet_paths = fetch_weights(Marigold.get_weights_paths("marigold-lcm-v1-0"))
    marigold = Marigold("marigold-lcm-v1-0", denoising_steps=4, ensemble_size=1, processing_resolution=256,
                        name="marigold", dependencies=[])
    marigold.vre_setup(load_weights=True)
    lib_unet = diffusers.UNet2DConditionModel(**UNET_CFG).eval()
    lib_unet.load_state_dict(vre_load_weights(unet_paths))
    lib_vae = diffusers.AutoencoderKL(**VAE_CFG).eval()
    lib_vae.load_state_dict(tr.load(vae_path, map_location="cpu"))
    lib_scheduler = diffusers.LCMScheduler(**LCM_CFG)
    assert sorted(lib_unet.state_dict()) == sorted(marigold.model.unet.state_dict())
    assert sorted(lib_vae.state_dict()) == sorted(marigold.model.vae.state_dict())

    rgb = tr.from_numpy(image_read(get_project_root() / "resources/demo1.jpg")).permute(2, 0, 1)[None].float()
    rgb = F.interpolate(rgb, size=(192, 256), mode="bilinear") / 255 * 2 - 1
    ctx = marigold.model.empty_text_embed.repeat(1, 1, 1)
    noise = tr.randn(1, 8, 24, 32, generator=tr.Generator().manual_seed(0))

    with tr.no_grad():
        # the UNet at a high and a low noise level, the VAE both ways
        for t in [tr.tensor(999), tr.tensor(259)]:
            y_lib = lib_unet(noise, t, encoder_hidden_states=ctx).sample
            y_vre = marigold.model.unet(noise, t, ctx)
            assert tr.equal(y_lib, y_vre), (t, (y_lib - y_vre).abs().max())
        latent_lib = lib_vae.quant_conv(lib_vae.encoder(rgb))
        latent_vre = marigold.model.vae.quant_conv(marigold.model.vae.encoder(rgb))
        assert tr.equal(latent_lib, latent_vre), (latent_lib - latent_vre).abs().max()
        img_lib = lib_vae.decoder(lib_vae.post_quant_conv(noise[:, 4:]))
        img_vre = marigold.model.vae.decoder(marigold.model.vae.post_quant_conv(noise[:, 4:]))
        assert tr.equal(img_lib, img_vre), (img_lib - img_vre).abs().max()

        # the whole seeded LCM denoising (4 steps): diffusers' modules in Marigold's loop vs VRE's single_infer
        gen = tr.Generator().manual_seed(42)
        rgb_latent = latent_lib.chunk(2, dim=1)[0] * 0.18215
        depth_latent = tr.randn(rgb_latent.shape, generator=gen)
        lib_scheduler.set_timesteps(4)
        for t in lib_scheduler.timesteps:
            noise_pred = lib_unet(tr.cat([rgb_latent, depth_latent], dim=1), t, encoder_hidden_states=ctx).sample
            depth_latent = lib_scheduler.step(noise_pred, t, depth_latent, generator=gen).prev_sample
        depth_lib = lib_vae.decoder(lib_vae.post_quant_conv(depth_latent / 0.18215)).mean(dim=1, keepdim=True)
        depth_lib = (depth_lib.clip(-1, 1) + 1) / 2
        depth_vre = marigold.model.single_infer(rgb, num_inference_steps=4, generator=tr.Generator().manual_seed(42))
    assert tr.equal(depth_lib, depth_vre), (depth_lib - depth_vre).abs().max()
