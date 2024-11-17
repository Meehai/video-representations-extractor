#!/usr/bin/env python3
"""
Marigold VRE representation
Standaline usage: `./marigold.py input/ output/` where `input/` is a directory of png/jpg/jpeg images
"""
import sys
from pathlib import Path
import numpy as np
import torch as tr
from overrides import overrides
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, DDIMScheduler, LCMScheduler, UNet2DConditionModel

from vre.utils import image_read, colorize_depth, image_write, fetch_weights, vre_load_weights, VREVideo, MemoryData
from vre.representations import (
    Representation, ReprOut, LearnedRepresentationMixin, ComputeRepresentationMixin, NpIORepresentation)
from vre.representations.depth.marigold.marigold_impl import MarigoldPipeline

class Marigold(Representation, LearnedRepresentationMixin, ComputeRepresentationMixin, NpIORepresentation):
    """Marigold VRE implementation"""
    def __init__(self, variant: str, denoising_steps: int, ensemble_size: int, processing_resolution: int,
                 seed: int | None = None, **kwargs):
        Representation.__init__(self, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        assert variant in ("marigold-v1-0", "marigold-lcm-v1-0", "testing"), variant
        self.variant = variant
        self.denoising_steps = denoising_steps
        self.ensemble_size = ensemble_size
        self.processing_resolution = processing_resolution
        self.model: MarigoldPipeline | None = None
        self.seed = seed

    @overrides
    def vre_setup(self, load_weights: bool=True):
        assert self.setup_called is False
        unet = UNet2DConditionModel(**self._get_unet_cfg())
        vae = AutoencoderKL(**self._get_vae_cfg())
        if load_weights:
            assert self.variant != "testing"
            vae.load_state_dict(vre_load_weights(fetch_weights(__file__) / "vae.pt")) # VAE is common for both variants
            unet.load_state_dict(vre_load_weights(fetch_weights(__file__) / f"{self.variant}_unet.pt"))

        if self.variant == "marigold-lcm-v1-0":
            lcm_scheduler_config = {**self._get_ddim_cfg(), "original_inference_steps": 50, "timestep_scaling": 10.0}
            scheduler = LCMScheduler(**lcm_scheduler_config)
        else:
            scheduler = DDIMScheduler(**self._get_ddim_cfg())
        self.model = MarigoldPipeline(unet=unet, vae=vae, scheduler=scheduler,
                                      scale_invariant=True, shift_invariant=True)
        self.model = self.model.to(self.device)
        self.setup_called = True

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        self.data = ReprOut(frames=video[ixs], key=ixs,
                            output=MemoryData([self._make_one_frame(frame) for frame in video[ixs]]))

    @overrides
    def make_images(self) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        return (colorize_depth(self.data.output, 0, 1) * 255).astype(np.uint8)

    @overrides
    def vre_free(self):
        assert self.setup_called is True and self.model is not None, (self.setup_called, self.model is not None)
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
        self.model = None
        self.setup_called = False

    @tr.no_grad
    def _make_one_frame(self, frame: np.ndarray):
        assert self.model is not None
        tr_rgb = tr.from_numpy(frame).permute(2, 0, 1)[None].to(self.device)
        generator = None
        if self.seed is not None:
            generator = tr.Generator(self.device)
            generator.manual_seed(self.seed)
        return self.model(tr_rgb, denoising_steps=self.denoising_steps, ensemble_size=self.ensemble_size,
                          processing_res=self.processing_resolution, generator=generator)[0]

    def _get_ddim_cfg(self) -> dict:
        return {
            "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085,
            "clip_sample": False, "clip_sample_range": 1.0,
            "dynamic_thresholding_ratio": 0.995, "num_train_timesteps": 1000,
            "prediction_type": "v_prediction", "rescale_betas_zero_snr": False,
            "sample_max_value": 1.0, "set_alpha_to_one": False, "steps_offset": 1,
            "thresholding": False, "timestep_spacing": "leading", "trained_betas": None
        }

    def _get_unet_cfg(self) -> dict:
        block_out_channels = {
            "testing": [8, 16, 32, 32],
            "marigold-lcm-v1-0": [320, 640, 1280, 1280],
            "marigold-v1-0": [320, 640, 1280, 1280],
        }[self.variant]
        norm_num_groups = {
            "testing": 8,
            "marigold-lcm-v1-0": 32,
            "marigold-v1-0": 32,
        }[self.variant]
        cfg = {
            "sample_size": 96, "in_channels": 8, "out_channels": 4, "center_input_sample": False,
            "flip_sin_to_cos": True, "freq_shift": 0,
            "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
            "mid_block_type": "UNetMidBlock2DCrossAttn",
            "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
            "only_cross_attention": False, "block_out_channels": block_out_channels, "layers_per_block": 2,
            "downsample_padding": 1, "mid_block_scale_factor": 1, "dropout": 0.0, "act_fn": "silu",
            "norm_num_groups": norm_num_groups, "norm_eps": 1e-05, "cross_attention_dim": 1024,
            "transformer_layers_per_block": 1, "reverse_transformer_layers_per_block": None,
            "encoder_hid_dim": None, "encoder_hid_dim_type": None, "attention_head_dim": [5, 10, 20, 20],
            "num_attention_heads": None, "dual_cross_attention": False, "use_linear_projection": True,
            "class_embed_type": None, "addition_embed_type": None, "addition_time_embed_dim": None,
            "num_class_embeds": None, "upcast_attention": False, "resnet_time_scale_shift": "default",
            "resnet_skip_time_act": False, "resnet_out_scale_factor": 1.0, "time_embedding_type": "positional",
            "time_embedding_dim": None, "time_embedding_act_fn": None, "timestep_post_act": None,
            "time_cond_proj_dim": None, "conv_in_kernel": 3, "conv_out_kernel": 3,
            "projection_class_embeddings_input_dim": None, "attention_type": "default",
            "class_embeddings_concat": False, "mid_block_only_cross_attention": None, "cross_attention_norm": None,
            "addition_embed_type_num_heads": 64
        }
        return cfg

    def _get_vae_cfg(self):
        block_out_channels = {
            "testing": [8, 16, 32, 32],
            "marigold-lcm-v1-0": [128, 256, 512, 512],
            "marigold-v1-0": [128, 256, 512, 512],
        }[self.variant]
        norm_num_groups = {
            "testing": 8,
            "marigold-lcm-v1-0": 32,
            "marigold-v1-0": 32,
        }[self.variant]

        return {
            "in_channels": 3, "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D",
            "DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            "block_out_channels": block_out_channels, "layers_per_block": 2, "act_fn": "silu",
            "latent_channels": 4, "norm_num_groups": norm_num_groups, "sample_size": 768,
            "scaling_factor": 0.18215, "shift_factor": None, "latents_mean": None,
            "latents_std": None, "force_upcast": True, "use_quant_conv": True,
            "use_post_quant_conv": True, "mid_block_add_attention": True
        }

def main():
    """main fn"""
    input_dir = Path(sys.argv[1])
    variant = "marigold-lcm-v1-0" if len(sys.argv) == 3 else sys.argv[3]
    device = "cuda" if tr.cuda.is_available() else "cpu"
    output_dir = Path(sys.argv[2]) / f"{variant}/{device}"
    image_paths = [x for x in input_dir.iterdir() if x.suffix in [".jpg", ".jpeg", ".png"]]
    assert len(image_paths) > 0, f"{input_dir} is empty of supported images"
    Path(output_dir / "png").mkdir(exist_ok=True, parents=True)

    marigold = Marigold(variant=variant, denoising_steps=4 if variant=="marigold-lcm-v1-0" else 10,
                        ensemble_size=1, processing_resolution=768, name="marigold", dependencies=[], seed=42)
    marigold.device = device
    marigold.vre_setup()

    expected = {
        "5131.png": {"cpu": (0.34113926, 0.25890285), "cuda": (0.37277249, 0.25874364)},
        "demo1.jpg": {"cpu": (0.47739768, 0.28376475), "cuda": (0.46834466, 0.32319403)},
        "demo2.jpg": {"cpu": (0.4603674, 0.2731725), "cuda": (0.51732534, 0.27054304)}
    }

    for rgb_path in (pbar := tqdm(image_paths, leave=True)):
        pbar.set_description(f"Estimating depth: {rgb_path.name}")
        rgb = image_read(rgb_path, "PIL")
        marigold.compute(rgb[None], [0])
        depth = marigold.data
        if marigold.variant == "marigold-lcm-v1-0" and rgb_path.name in expected.keys() \
                and marigold.seed is not None and marigold.seed == 42 and str(device) in expected[rgb_path.name]:
            try:
                assert np.allclose(A := expected[rgb_path.name][str(device)][0], (B := depth.output[0].mean())), (A, B)
                assert np.allclose(A := expected[rgb_path.name][str(device)][1], (B := depth.output[0].std())), (A, B)
                print("Mean and std match!")
            except AssertionError as e:
                print(e)

        marigold.data = marigold.resize(marigold.data, rgb.shape[0:2])
        depth_img = marigold.make_images()[0]
        image_write(depth_img, output_dir / "png" / f"{rgb_path.stem}_pred.png")

if __name__ == "__main__":
    main()
