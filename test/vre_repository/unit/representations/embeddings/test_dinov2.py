import numpy as np
import pytest
import torch as tr
from vre_video import VREVideo
from vre import ReprOut, MemoryData
from vre_repository.embeddings.dinov2 import DinoV2
from vre_repository.embeddings.dinov2.dinov2_impl import Dinov2Model

def test_dinov2():
    video = VREVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    dinov2_repr = DinoV2("testing", pool="cat", long_side=70, name="dinov2", dependencies=[])
    dinov2_repr.vre_setup(load_weights=False)
    assert dinov2_repr.name == "dinov2"
    assert dinov2_repr.compress is True # default
    assert dinov2_repr.device == "cpu" # default from LearnedRepresentationMixin
    assert dinov2_repr.n_channels == 64 # cat = 2 * hidden_size (32)

    out = dinov2_repr.compute(video, ixs=[0, 1])
    assert out.output.shape == (2, 64) and out.output.dtype == np.float32, (out.output.shape, out.output.dtype)
    assert np.allclose(np.linalg.norm(out.output, axis=1), 1, atol=1e-5) # L2-normalized
    assert np.allclose(dinov2_repr.compute(video, ixs=[0, 1]).output, out.output) # eval mode: no randomness

    out_images = dinov2_repr.make_images(out)
    assert out_images.shape == (2, 64, 128, 3) # the frame's size, so it fits in a collage next to rgb
    assert out_images.dtype == np.uint8

    assert dinov2_repr.size(out) == (2, 64)
    out_resized = dinov2_repr.resize(out, (32, 32)) # a vector has no size: resize keeps it
    assert np.array_equal(out_resized.output, out.output)
    assert dinov2_repr.size(out_resized) == (2, 64)

    dinov2_repr.vre_free()
    assert dinov2_repr.setup_called is False and dinov2_repr.model is None

def test_dinov2_resize_output_images():
    video = VREVideo(np.random.randint(0, 255, size=(2, 64, 128, 3), dtype=np.uint8), fps=30)
    dinov2_repr = DinoV2("testing", pool="cls", name="dinov2", dependencies=[])
    dinov2_repr.vre_setup(load_weights=False)
    out = dinov2_repr.compute(video, ixs=[0, 1])
    out = ReprOut(frames=out.frames, output=out.output, key=out.key, output_images=dinov2_repr.make_images(out))
    out_resized = dinov2_repr.resize(out, (32, 48))
    assert out_resized.output_images.shape == (2, 32, 48, 3)
    assert np.array_equal(out_resized.output, out.output)

@pytest.mark.parametrize("pool, n_channels", [("cls", 32), ("gem", 32), ("cat", 64)])
def test_dinov2_pools(pool: str, n_channels: int):
    video = VREVideo(np.random.randint(0, 255, size=(2, 40, 30, 3), dtype=np.uint8), fps=30)
    dinov2_repr = DinoV2("testing", pool=pool, long_side=28, name="dinov2", dependencies=[])
    dinov2_repr.vre_setup(load_weights=False)
    assert dinov2_repr.n_channels == n_channels
    out = dinov2_repr.compute(video, ixs=[0, 1])
    assert out.output.shape == (2, n_channels)
    assert np.allclose(np.linalg.norm(out.output, axis=1), 1, atol=1e-5)

def test_dinov2_make_images_without_frames():
    dinov2_repr = DinoV2("testing", pool="cat", name="dinov2", dependencies=[]) # D = 64 -> an 8x8 grid
    out = ReprOut(frames=None, output=MemoryData(np.random.randn(3, 64).astype(np.float32)), key=[0, 1, 2])
    images = dinov2_repr.make_images(out)
    assert images.shape == (3, 8, 8, 3) and images.dtype == np.uint8 # vre_reader: no frames, the raw grid
    assert images.min() == 0 and images.max() == 255 # min-max scaled per frame

def test_dinov2_bad_args():
    with pytest.raises(ValueError):
        DinoV2("dinov2-huge", name="dinov2", dependencies=[])
    with pytest.raises(ValueError):
        DinoV2("testing", pool="mean", name="dinov2", dependencies=[])
    with pytest.raises(ValueError):
        DinoV2("testing", long_side=10, name="dinov2", dependencies=[])
    with pytest.raises(AssertionError):
        DinoV2("testing", name="dinov2", dependencies=[]).vre_setup(load_weights=True) # 'testing' has no weights

def test_dinov2_preprocess_shape():
    video = VREVideo(np.random.randint(0, 255, size=(1, 1080, 1920, 3), dtype=np.uint8), fps=30)
    dinov2_repr = DinoV2("testing", long_side=672, name="dinov2", dependencies=[])
    dinov2_repr.vre_setup(load_weights=False)
    x = dinov2_repr._preprocess(video[[0]])
    assert x.shape == (1, 3, 378, 672) # long side 672, aspect kept, both sides multiple of 14

def test_dinov2_model_pos_embeddings():
    model = Dinov2Model(hidden_size=32, num_layers=1, num_heads=2).eval()
    assert model.embeddings.position_embeddings.shape == (1, 37 * 37 + 1, 32) # trained at 518 = 37 * 14
    with tr.no_grad():
        assert model(tr.randn(2, 3, 518, 518)).shape == (2, 1 + 37 * 37, 32) # native grid: no interpolation
        assert model(tr.randn(2, 3, 28, 42)).shape == (2, 1 + 2 * 3, 32) # interpolated to a 2x3 grid
        assert model(tr.randn(1, 3, 30, 44)).shape == (1, 1 + 2 * 3, 32) # the patch conv floors H, W to a multiple

def test_dinov2_model_batch_independent():
    model = Dinov2Model(hidden_size=32, num_layers=2, num_heads=2).eval()
    x = tr.randn(3, 3, 42, 56)
    with tr.no_grad():
        y_batch = model(x)
        y_one = model(x[1:2])
    assert tr.allclose(y_batch[1:2], y_one, atol=1e-5) # each frame's embedding is independent of its batch

if __name__ == "__main__":
    test_dinov2()
