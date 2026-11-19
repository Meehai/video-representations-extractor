"""
Marigold's diffusion schedulers in plain torch: ports of diffusers' (0.30.3) DDIMScheduler and LCMScheduler limited to
the config Marigold uses: 1000 train steps, scaled_linear betas (0.00085 -> 0.012), v_prediction, no clipping or
thresholding, final alpha_cumprod = alphas_cumprod[0] (set_alpha_to_one=False). DDIM: eta=0, 'leading' spacing,
steps_offset=1. LCM: 50 original inference steps, timestep_scaling=10. Same numerics as diffusers.
"""
import numpy as np
import torch as tr

NUM_TRAIN_TIMESTEPS = 1000

class _Scheduler:
    """the shared noise schedule. alphas_cumprod stays on cpu (as in diffusers) and is indexed with the timesteps"""
    max_inference_steps = NUM_TRAIN_TIMESTEPS

    def __init__(self):
        betas = tr.linspace(0.00085**0.5, 0.012**0.5, NUM_TRAIN_TIMESTEPS, dtype=tr.float32) ** 2
        self.alphas_cumprod = tr.cumprod(1.0 - betas, dim=0)
        self.timesteps: tr.Tensor | None = None

    def set_timesteps(self, num_inference_steps: int, device: str | tr.device | None = None):
        """sets the `timesteps` (descending, distinct, int64) of the denoising loop"""
        if not 1 <= num_inference_steps <= self.max_inference_steps:
            raise ValueError(f"num_inference_steps={num_inference_steps} not in [1, {self.max_inference_steps}]")
        self.timesteps = tr.from_numpy(self._make_timesteps(num_inference_steps)).to(device=device, dtype=tr.long)

    def _make_timesteps(self, num_inference_steps: int) -> np.ndarray:
        raise NotImplementedError

class DDIMScheduler(_Scheduler):
    """deterministic DDIM (eta=0)"""
    def _make_timesteps(self, num_inference_steps: int) -> np.ndarray:
        step_ratio = NUM_TRAIN_TIMESTEPS // num_inference_steps
        return (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64) + 1

    def step(self, model_output: tr.Tensor, timestep: tr.Tensor, sample: tr.Tensor,
             generator: tr.Generator | None = None) -> tr.Tensor:
        """x_t -> x_{t-1} given the v-prediction. The generator is unused (eta=0: no noise is added)"""
        assert self.timesteps is not None, "call set_timesteps() first"
        _ = generator
        prev_timestep = timestep - NUM_TRAIN_TIMESTEPS // len(self.timesteps)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.alphas_cumprod[0]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        return alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

class LCMScheduler(_Scheduler):
    """latent consistency model multistep sampling: denoise, then re-noise (with the generator) to the next timestep"""
    max_inference_steps = 50 # original_inference_steps
    timestep_scaling = 10.0
    sigma_data = 0.5

    def _make_timesteps(self, num_inference_steps: int) -> np.ndarray:
        k = NUM_TRAIN_TIMESTEPS // self.max_inference_steps
        lcm_origin_timesteps = (np.arange(1, self.max_inference_steps + 1) * k - 1)[::-1].copy()
        inference_indices = np.linspace(0, len(lcm_origin_timesteps), num=num_inference_steps, endpoint=False)
        return lcm_origin_timesteps[np.floor(inference_indices).astype(np.int64)]

    def step(self, model_output: tr.Tensor, timestep: tr.Tensor, sample: tr.Tensor,
             generator: tr.Generator | None = None) -> tr.Tensor:
        """x_t -> x_{t_next} given the v-prediction. Draws noise from the generator except on the last step"""
        assert self.timesteps is not None, "call set_timesteps() first"
        ix = (self.timesteps == timestep.to(self.timesteps.device)).nonzero()
        assert len(ix) == 1, (timestep, self.timesteps)
        is_last = ix.item() == len(self.timesteps) - 1
        prev_timestep = timestep if is_last else self.timesteps[ix.item() + 1]
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        beta_prod_t, beta_prod_t_prev = 1 - alpha_prod_t, 1 - alpha_prod_t_prev

        scaled_timestep = timestep * self.timestep_scaling
        c_skip = self.sigma_data**2 / (scaled_timestep**2 + self.sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + self.sigma_data**2) ** 0.5
        predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        denoised = c_out * predicted_original_sample + c_skip * sample
        if is_last:
            return denoised
        noise = tr.randn(model_output.shape, generator=generator, device=model_output.device, dtype=denoised.dtype)
        return alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
