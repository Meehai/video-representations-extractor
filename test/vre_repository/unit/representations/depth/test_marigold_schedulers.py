"""Marigold's DDIM/LCM schedulers: the timesteps diffusers would pick, determinism, and the LCM denoising loop"""
import numpy as np
import pytest
import torch as tr
from vre_repository.depth.marigold import Marigold
from vre_repository.depth.marigold.marigold_impl import DDIMScheduler, LCMScheduler
from vre_video import VREVideo

def test_ddim_timesteps():
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(10)
    assert scheduler.timesteps.dtype == tr.long, scheduler.timesteps.dtype
    assert scheduler.timesteps.tolist() == [901, 801, 701, 601, 501, 401, 301, 201, 101, 1], scheduler.timesteps
    with pytest.raises(ValueError):
        scheduler.set_timesteps(0)

def test_lcm_timesteps():
    scheduler = LCMScheduler()
    scheduler.set_timesteps(4)
    assert scheduler.timesteps.dtype == tr.long, scheduler.timesteps.dtype
    assert scheduler.timesteps.tolist() == [999, 759, 499, 259], scheduler.timesteps
    scheduler.set_timesteps(50)
    assert scheduler.timesteps.tolist() == list(range(999, 0, -20)), scheduler.timesteps
    with pytest.raises(ValueError):
        scheduler.set_timesteps(51)

def test_ddim_step_ignores_generator():
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(4)
    x, v = tr.randn(2, 4, 5, 7), tr.randn(2, 4, 5, 7)
    y1 = scheduler.step(v, scheduler.timesteps[0], x, generator=tr.Generator().manual_seed(0))
    y2 = scheduler.step(v, scheduler.timesteps[0], x, generator=tr.Generator().manual_seed(1))
    assert y1.shape == x.shape and tr.equal(y1, y2)

def test_lcm_step_chain_seeded():
    scheduler = LCMScheduler()
    scheduler.set_timesteps(4)
    x, v = tr.randn(2, 4, 5, 7), tr.randn(2, 4, 5, 7)
    gen1, gen2 = tr.Generator().manual_seed(42), tr.Generator().manual_seed(42)
    y1 = y2 = x
    for t in scheduler.timesteps[:-1]:
        y1, y2 = scheduler.step(v, t, y1, generator=gen1), scheduler.step(v, t, y2, generator=gen2)
    assert tr.equal(y1, y2) and not tr.equal(y1, x)

    # the last step only denoises: no noise drawn, so the generator does not matter
    last1 = scheduler.step(v, scheduler.timesteps[-1], y1, generator=tr.Generator().manual_seed(0))
    last2 = scheduler.step(v, scheduler.timesteps[-1], y1, generator=tr.Generator().manual_seed(1))
    assert tr.equal(last1, last2)

    # any other step re-noises: different generators, different outputs
    mid1 = scheduler.step(v, scheduler.timesteps[1], x, generator=tr.Generator().manual_seed(0))
    mid2 = scheduler.step(v, scheduler.timesteps[1], x, generator=tr.Generator().manual_seed(1))
    assert not tr.equal(mid1, mid2)

def test_marigold_testing_lcm():
    video = VREVideo(np.random.randint(0, 255, size=(2, 64, 128, 3), dtype=np.uint8), fps=30)
    marigold_repr = Marigold("testing", denoising_steps=4, ensemble_size=1, processing_resolution=30, seed=0,
                             name="marigold", dependencies=[])
    marigold_repr.vre_setup(load_weights=False)
    marigold_repr.model.diffusion_scheduler = LCMScheduler() # 'testing' uses DDIM; the shipped variant uses LCM

    out1 = marigold_repr.compute(video, ixs=[0])
    out2 = marigold_repr.compute(video, ixs=[0])
    assert out1.output.shape == (1, 8, 24, 1), out1.output.shape
    assert np.isfinite(out1.output).all()
    assert np.array_equal(out1.output, out2.output) # same seed => same output
