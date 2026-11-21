from pathlib import Path
from urllib.error import HTTPError
import numpy as np
import pytest
import torch as tr
from torch.nn import functional as F
from vre_video import VREVideo
from vre import ReprOut, MemoryData
from vre_repository.embeddings.dinov2 import DinoV2
from vre_repository.embeddings.dinov2 import dinov2 as dinov2_module
from vre_repository.embeddings.dinov2.dinov2_impl import Dinov2Model

def test_dinov2():
    video = VREVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    dinov2_repr = DinoV2("testing", pool="mean", input_size=(42, 70), name="dinov2", dependencies=[])
    dinov2_repr.vre_setup(load_weights=False)
    assert dinov2_repr.name == "dinov2"
    assert dinov2_repr.compress is True # default
    assert dinov2_repr.device == "cpu" # default from LearnedRepresentationMixin
    assert dinov2_repr.n_channels == 32 # hidden_size

    out = dinov2_repr.compute(video, ixs=[0, 1])
    assert out.output.shape == (2, 32) and out.output.dtype == np.float32, (out.output.shape, out.output.dtype)
    assert np.allclose(np.linalg.norm(out.output, axis=1), 1, atol=1e-5) # L2-normalized
    assert np.allclose(dinov2_repr.compute(video, ixs=[0, 1]).output, out.output) # eval mode: no randomness

    out_images = dinov2_repr.make_images(out)
    assert out_images.shape == (2, 64, 128, 3) # the frame's size, so it fits in a collage next to rgb
    assert out_images.dtype == np.uint8

    assert dinov2_repr.size(out) == (2, 32)
    out_resized = dinov2_repr.resize(out, (32, 32)) # a vector has no size: resize keeps it
    assert np.array_equal(out_resized.output, out.output)
    assert dinov2_repr.size(out_resized) == (2, 32)

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

@pytest.mark.parametrize("pool", ["cls", "gem", "mean"])
def test_dinov2_pools(pool: str):
    video = VREVideo(np.random.randint(0, 255, size=(2, 40, 30, 3), dtype=np.uint8), fps=30)
    dinov2_repr = DinoV2("testing", pool=pool, input_size=(28, 28), name="dinov2", dependencies=[])
    dinov2_repr.vre_setup(load_weights=False)
    assert dinov2_repr.n_channels == 32 # hidden_size, whatever the pool
    out = dinov2_repr.compute(video, ixs=[0, 1])
    assert out.output.shape == (2, 32)
    assert np.allclose(np.linalg.norm(out.output, axis=1), 1, atol=1e-5)

def test_dinov2_pools_from_tokens():
    video = VREVideo(np.random.randint(0, 255, size=(2, 40, 30, 3), dtype=np.uint8), fps=30)
    dinov2_cls = DinoV2("testing", pool="cls", input_size=(28, 28), name="dinov2", dependencies=[])
    dinov2_gem = DinoV2("testing", pool="gem", input_size=(28, 28), name="dinov2", dependencies=[]) # gem_p=3 default
    dinov2_mean = DinoV2("testing", pool="mean", input_size=(28, 28), name="dinov2", dependencies=[])
    dinov2_cls.vre_setup(load_weights=False)
    dinov2_gem.vre_setup(load_weights=False)
    dinov2_mean.vre_setup(load_weights=False)
    dinov2_gem.model.load_state_dict(dinov2_cls.model.state_dict()) # same weights: only the pool differs
    dinov2_mean.model.load_state_dict(dinov2_cls.model.state_dict())
    with tr.inference_mode():
        tokens = dinov2_cls.model(dinov2_cls._preprocess(video[[0, 1]])) # (2, 1 + 4, 32): CLS + 2x2 patches
    cls = tokens[:, 0]
    gem = tokens[:, 1:].clamp(min=1e-6).pow(3).mean(dim=1).pow(1 / 3) # GeM over the patches, p=3
    mean = tokens[:, 1:].mean(dim=1) # plain average over the patches, CLS excluded
    assert np.allclose(dinov2_cls.compute(video, ixs=[0, 1]).output, F.normalize(cls, dim=-1).numpy(), atol=1e-6)
    assert np.allclose(dinov2_gem.compute(video, ixs=[0, 1]).output, F.normalize(gem, dim=-1).numpy(), atol=1e-6)
    assert np.allclose(dinov2_mean.compute(video, ixs=[0, 1]).output, F.normalize(mean, dim=-1).numpy(), atol=1e-6)

def test_dinov2_gem_p():
    video = VREVideo(np.random.randint(0, 255, size=(2, 40, 30, 3), dtype=np.uint8), fps=30)
    dinov2_p1 = DinoV2("testing", pool="gem", gem_p=1, input_size=(28, 28), name="dinov2", dependencies=[])
    dinov2_p3 = DinoV2("testing", pool="gem", gem_p=3, input_size=(28, 28), name="dinov2", dependencies=[])
    dinov2_mean = DinoV2("testing", pool="mean", input_size=(28, 28), name="dinov2", dependencies=[])
    dinov2_p1.vre_setup(load_weights=False)
    dinov2_p3.vre_setup(load_weights=False)
    dinov2_mean.vre_setup(load_weights=False)
    dinov2_p3.model.load_state_dict(dinov2_p1.model.state_dict()) # same weights: only the pool differs
    dinov2_mean.model.load_state_dict(dinov2_p1.model.state_dict())
    with tr.inference_mode():
        tokens = dinov2_p1.model(dinov2_p1._preprocess(video[[0, 1]]))
    y_p1 = dinov2_p1.compute(video, ixs=[0, 1]).output
    assert np.allclose(y_p1, F.normalize(tokens[:, 1:].clamp(min=1e-6).mean(dim=1), dim=-1).numpy(), atol=1e-6)
    assert not np.allclose(y_p1, dinov2_p3.compute(video, ixs=[0, 1]).output, atol=1e-4) # p is used
    # tokens are signed (post-layernorm) and GeM clamps them at 0: p=1 is not 'mean'
    assert not np.allclose(y_p1, dinov2_mean.compute(video, ixs=[0, 1]).output, atol=1e-4)

def test_dinov2_make_images_without_frames():
    dinov2_repr = DinoV2("testing", pool="gem", name="dinov2", dependencies=[]) # D = 32 -> a 4x8 grid
    out = ReprOut(frames=None, output=MemoryData(np.random.randn(3, 32).astype(np.float32)), key=[0, 1, 2])
    images = dinov2_repr.make_images(out)
    assert images.shape == (3, 4, 8, 3) and images.dtype == np.uint8 # vre_reader: no frames, the raw grid
    assert images.min() == 0 and images.max() == 255 # min-max scaled per frame

def test_dinov2_tokens():
    video = VREVideo(np.random.randint(0, 255, size=(2, 64, 128, 3), dtype=np.uint8), fps=30)
    dinov2_repr = DinoV2("testing", pool="none", input_size=(28, 42), name="dinov2", dependencies=[])
    dinov2_repr.vre_setup(load_weights=False)
    assert dinov2_repr.n_channels == 32 # hidden_size: one D-vector per patch

    out = dinov2_repr.compute(video, ixs=[0, 1])
    assert out.output.shape == (2, 2, 3, 32) and out.output.dtype == np.float32 # (28/14, 42/14) patch grid
    assert not np.allclose(np.linalg.norm(out.output, axis=-1), 1, atol=1e-3) # raw tokens: not L2-normalized

    with tr.inference_mode():
        tokens = dinov2_repr.model(dinov2_repr._preprocess(video[[0, 1]])).numpy()
    assert np.allclose(out.output.reshape(2, 6, 32), tokens[:, 1:], atol=1e-6) # patch tokens, row-major, no CLS

    out_images = dinov2_repr.make_images(out)
    assert out_images.shape == (2, 64, 128, 3) and out_images.dtype == np.uint8 # the frame's size
    assert out_images.min() == 0 and out_images.max() == 255 # top-3 PCA components, min-max scaled per frame
    assert np.array_equal(dinov2_repr.make_images(out), out_images) # deterministic: PCA signs are fixed

    out_resized = dinov2_repr.resize(out, (32, 32)) # the grid is set by input_size, not output_size
    assert np.array_equal(out_resized.output, out.output)

def test_dinov2_tokens_make_images_without_frames():
    dinov2_repr = DinoV2("testing", pool="none", name="dinov2", dependencies=[])
    out = ReprOut(frames=None, output=MemoryData(np.random.randn(3, 4, 5, 32).astype(np.float32)), key=[0, 1, 2])
    images = dinov2_repr.make_images(out)
    assert images.shape == (3, 4, 5, 3) and images.dtype == np.uint8 # vre_reader: no frames, the raw grid

def test_dinov2_tokens_make_images_pca():
    dinov2_repr = DinoV2("testing", pool="none", name="dinov2", dependencies=[])
    grid = np.zeros((1, 2, 4, 32), dtype=np.float32)
    grid[0, :, 0:2] = np.random.randn(32) # left half: one token, repeated
    grid[0, :, 2:4] = np.random.randn(32) # right half: another one
    out = ReprOut(frames=None, output=MemoryData(grid), key=[0])
    image = dinov2_repr.make_images(out)[0]
    assert (image[:, 0:2] == image[0, 0]).all() and (image[:, 2:4] == image[0, 2]).all() # same token, same color
    assert not np.array_equal(image[0, 0], image[0, 2]) # different tokens, different colors

def test_dinov2_tokens_make_images_tiny_grid():
    dinov2_repr = DinoV2("testing", pool="none", name="dinov2", dependencies=[])
    out = ReprOut(frames=None, output=MemoryData(np.random.randn(2, 1, 2, 32).astype(np.float32)), key=[0, 1])
    images = dinov2_repr.make_images(out) # 2 tokens: rank 1 once centered, so 1 component; the other 2 channels are 0
    assert images.shape == (2, 1, 2, 3) and images.dtype == np.uint8
    assert (images[..., 1:] == 0).all()

def test_dinov2_input_size_changes_output():
    video = VREVideo(np.random.randint(0, 255, size=(1, 64, 128, 3), dtype=np.uint8), fps=30)
    dinov2_small = DinoV2("testing", pool="gem", input_size=(28, 56), name="dinov2", dependencies=[])
    dinov2_big = DinoV2("testing", pool="gem", input_size=(56, 112), name="dinov2", dependencies=[])
    dinov2_small.vre_setup(load_weights=False)
    dinov2_big.vre_setup(load_weights=False)
    dinov2_big.model.load_state_dict(dinov2_small.model.state_dict()) # same weights, only input_size differs
    y_small = dinov2_small.compute(video, ixs=[0]).output
    y_big = dinov2_big.compute(video, ixs=[0]).output
    assert y_small.shape == y_big.shape == (1, 32) # pooled: same shape
    assert not np.allclose(y_small, y_big, atol=1e-4) # but input_size is part of what the vectors mean

def test_dinov2_bad_args():
    with pytest.raises(ValueError):
        DinoV2("dinov2-huge", name="dinov2", dependencies=[])
    with pytest.raises(ValueError):
        DinoV2("testing", pool="cat", name="dinov2", dependencies=[])
    with pytest.raises(ValueError):
        DinoV2("testing", pool="gem", gem_p=0, name="dinov2", dependencies=[]) # 1/p: p must be > 0
    with pytest.raises(ValueError):
        DinoV2("testing", input_size=(10, 14), name="dinov2", dependencies=[]) # smaller than a patch
    with pytest.raises(ValueError):
        DinoV2("testing", input_size=(14, 30), name="dinov2", dependencies=[]) # not a multiple of 14: no rounding
    with pytest.raises(ValueError):
        DinoV2("testing", input_size=(14, ), name="dinov2", dependencies=[]) # (h, w) only
    with pytest.raises(AssertionError):
        DinoV2("testing", name="dinov2", dependencies=[]).vre_setup(load_weights=True) # 'testing' has no weights

def test_dinov2_preprocess_shape():
    video = VREVideo(np.random.randint(0, 255, size=(1, 1080, 1920, 3), dtype=np.uint8), fps=30)
    dinov2_repr = DinoV2("testing", name="dinov2", dependencies=[])
    dinov2_repr.vre_setup(load_weights=False)
    assert dinov2_repr._preprocess(video[[0]]).shape == (1, 3, 378, 672) # default input_size: 16:9, long side 672
    dinov2_repr = DinoV2("testing", input_size=[560, 560], name="dinov2", dependencies=[]) # a list, as yaml gives it
    dinov2_repr.vre_setup(load_weights=False)
    assert dinov2_repr.input_size == (560, 560)
    assert dinov2_repr._preprocess(video[[0]]).shape == (1, 3, 560, 560) # exactly input_size, aspect not kept

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

def test_dinov2_weights_fallback_to_huggingface(monkeypatch: pytest.MonkeyPatch):
    def gitlab_404(paths: list[Path]):
        raise HTTPError(url=str(paths[0]), code=404, msg="Not Found", hdrs=None, fp=None)
    fetched = []
    monkeypatch.setattr(dinov2_module, "fetch_weights", gitlab_404) # stub the network: not on VRE's gitlab
    monkeypatch.setattr(dinov2_module, "fetch", lambda src, dst: fetched.append((src, dst)) or dst)
    dinov2_repr = DinoV2("dinov2-base", name="dinov2", dependencies=[])

    path = dinov2_repr._fetch_weights()
    assert path == DinoV2.get_weights_paths("dinov2-base")[0] # same destination as VRE's own weights
    assert fetched == [("https://huggingface.co/facebook/dinov2-base/resolve/"
                        "f9e44c814b77203eaa57a6bdbbd535f21ede1415/pytorch_model.bin", path)] # pinned commit

def test_dinov2_weights_no_fallback_on_other_errors(monkeypatch: pytest.MonkeyPatch):
    def broken_file(paths: list[Path]):
        raise AssertionError(f"'{paths[0]}' exists (or is a dir)")
    monkeypatch.setattr(dinov2_module, "fetch_weights", broken_file)
    monkeypatch.setattr(dinov2_module, "fetch", lambda src, dst: pytest.fail("huggingface must not be tried"))
    with pytest.raises(AssertionError): # only network errors fall back: a local problem must surface as-is
        DinoV2("dinov2-small", name="dinov2", dependencies=[])._fetch_weights()

if __name__ == "__main__":
    test_dinov2()
