"""marigold pipeline impl"""
import logging
from typing import Dict, Optional, Union
import os

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, LCMScheduler, UNet2DConditionModel
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .ensemble import ensemble_depth
from .marigold_util import resize_max_res, find_batch_size


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(unet=unet, vae=vae, scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer)
        self.register_to_config(scale_invariant=scale_invariant, shift_invariant=shift_invariant,
                                default_denoising_steps=default_denoising_steps,
                                default_processing_resolution=default_processing_resolution)

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution
        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        rgb: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        ensemble_kwargs: Dict = None,
    ) -> np.ndarray:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution
        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)
        assert 4 == rgb.dim() and 3 == rgb.shape[-3], f"Wrong input shape {rgb.shape}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(rgb, max_edge_resolution=processing_res, resample_method=InterpolationMode.BILINEAR)

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(ensemble_size=ensemble_size, input_res=max(rgb_norm.shape[1:]), dtype=self.dtype)

        # Predict depth maps (batched)
        depth_pred_ls = []
        iterable = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)
        for batch in iterable:
            (batched_img,) = batch
            depth_pred_raw = self.single_infer(rgb_in=batched_img, num_inference_steps=denoising_steps,
                                               generator=generator)
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depth(
                depth_preds,
                scale_invariant=self.scale_invariant,
                shift_invariant=self.shift_invariant,
                max_res=50,
                **(ensemble_kwargs or {}),
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None

        assert pred_uncert is None, "Ensembles not supported yet"
        return depth_pred.squeeze().cpu().numpy()

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")


    @torch.no_grad()
    def single_infer(self, rgb_in: torch.Tensor, num_inference_steps: int,
                     generator: Union[torch.Generator, None]) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # Initial depth map (noise)
        depth_latent = torch.randn(rgb_latent.shape, device=device, dtype=self.dtype, generator=generator)# [B, 4, h, w]

        # Batched empty text embedding TODO: can we remove this ??
        if self.empty_text_embed is None:
            text_inputs = self.tokenizer("", padding="do_not_pad", max_length=self.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
            self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
        batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1)).to(device)  # [B, 2, 1024]

        # Denoising loop
        for t in tqdm(timesteps, total=len(timesteps), leave=False,
                      disable=os.getenv("MARIGOLD_PBAR", "0") == "0", desc=" " * 4 + "Diffusion denoising"):
            unet_input = torch.cat([rgb_latent, depth_latent], dim=1)  # this order is important
            # predict the noise residual
            noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample  # [B, 4, h, w]
            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent, generator=generator).prev_sample

        depth = self.decode_depth(depth_latent)

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """Encode RGB image into latent."""
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.rgb_latent_scale_factor # scale latent
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """Decode depth latent into depth map."""
        depth_latent = depth_latent / self.depth_latent_scale_factor # scale latent
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z) # decode
        depth_mean = stacked.mean(dim=1, keepdim=True) # mean of output channels
        return depth_mean
