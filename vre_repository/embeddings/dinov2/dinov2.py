"""DINOv2 image embeddings: global vectors or patch tokens. Weights: transformers' facebook/dinov2-* (see README.md)."""
from pathlib import Path
from urllib.error import URLError
import numpy as np
import torch as tr
from torch.nn import functional as F
from overrides import overrides

from vre_video import VREVideo
from vre.utils import MemoryData, fetch
from vre.logger import vre_logger as logger
from vre import ReprOut
from vre.representations.mixins import LearnedRepresentationMixin
from vre_repository.utils import fetch_weights
from vre_repository.vre_image_utils import image_resize_batch
from vre_repository.embeddings import EmbeddingRepresentation

from .dinov2_impl import Dinov2Model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VARIANTS = { # variant: (hidden_size, num_layers, num_heads)
    "dinov2-small": (384, 12, 6),
    "dinov2-base": (768, 12, 12),
    "dinov2-large": (1024, 24, 16),
    "testing": (32, 2, 2),
}
# Fallback when VRE's gitlab doesn't have weights/<variant>.pt: huggingface's facebook/<variant> pytorch_model.bin at
# a pinned commit, which is the very same state_dict (same keys, same tensors).
HF_URL = "https://huggingface.co/facebook/{variant}/resolve/{revision}/pytorch_model.bin"
HF_REVISIONS = {
    "dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "dinov2-base": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    "dinov2-large": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
}
POOLS = ("cls", "gem", "mean", "none")
PATCH_SIZE = 14

class DinoV2(EmbeddingRepresentation, LearnedRepresentationMixin):
    """
    DINOv2 embeddings per frame: one L2-normalized vector, or the raw patch tokens (pool='none').
    - variant: one of VARIANTS ('testing' = tiny random model, no weights).
    - pool: 'cls' = the CLS token, 'gem' = GeM (power gem_) over the patch tokens, 'mean' = avg, each L2-normed.
      'none' = no pooling: the patch tokens as a (h/14, w/14, D) grid, not normalized (input for dense matchers).
    - input_size: (h, w) every frame is resized to before the model, both multiples of 14 (the patch size). It sets
      the output: pool='none' gives a (h/14, w/14) token grid, and the pooled vectors change with it too.
    On cuda the model runs under float16 autocast.
    """
    def __init__(self, variant: str, pool: str = "cls", input_size: tuple[int, int] = (378, 672),
                 gem_p: float = 3, **kwargs):
        EmbeddingRepresentation.__init__(self, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        if variant not in VARIANTS:
            raise ValueError(f"{variant=} not in {list(VARIANTS)}")
        if pool not in POOLS:
            raise ValueError(f"{pool=} not in {POOLS}")
        if len(input_size) != 2 or any(s < PATCH_SIZE or s % PATCH_SIZE != 0 for s in input_size):
            raise ValueError(f"{input_size=} must be (h, w), each a positive multiple of the patch size ({PATCH_SIZE})")
        if gem_p <= 0:
            raise ValueError(f"{gem_p=} cannot be <=0")
        self.variant = variant
        self.pool = pool
        self.gem_p = gem_p
        self.input_size = tuple(input_size) # yaml gives a list
        self.model: Dinov2Model | None = None

    @property
    @overrides
    def n_channels(self) -> int:
        return VARIANTS[self.variant][0]

    @overrides
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        assert self.setup_called is True and self.model is not None, (self.setup_called, self.model is not None)
        frames = video[ixs]
        with tr.inference_mode():
            x = self._preprocess(frames)
            with tr.autocast(device_type="cuda", dtype=tr.float16, enabled=str(self.device).startswith("cuda")):
                tokens = self.model(x)
            if self.pool == "none":
                grid_h, grid_w = x.shape[2] // PATCH_SIZE, x.shape[3] // PATCH_SIZE
                embeddings = tokens[:, 1:].float().reshape(len(frames), grid_h, grid_w, -1).cpu().numpy()
            else:
                embeddings = F.normalize(self._pool(tokens.float()), dim=-1).cpu().numpy()
        return ReprOut(frames=frames, output=MemoryData(embeddings), key=ixs)

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        """pool='none': each frame's token grid as RGB = its top-3 principal components (see _pca_rgb)"""
        if self.pool != "none":
            return super().make_images(data)
        emb = np.asarray(data.output)
        assert emb.ndim == 4 and emb.shape[-1] == self.n_channels, (emb.shape, self.n_channels)
        images = np.stack([_pca_rgb(grid) for grid in emb])
        if data.frames is None: # vre_reader: no frames, so the grid stays at its own size
            return images
        return image_resize_batch(images, *data.frames[0].shape[0:2], interpolation="nearest")

    @staticmethod
    @overrides
    def get_weights_paths(variant: str | None = None) -> list[str]:
        assert variant in VARIANTS and variant != "testing", variant
        return [Path(__file__).parent / f"weights/{variant}.pt"]

    @overrides
    def vre_setup(self, load_weights: bool = True):
        assert self.setup_called is False
        assert load_weights is False or self.variant != "testing", "the 'testing' variant has no weights"
        self.model = Dinov2Model(*VARIANTS[self.variant])
        if load_weights:
            self.model.load_state_dict(tr.load(self._fetch_weights(), map_location="cpu"))
        self.model = self.model.eval().to(self.device)
        self.setup_called = True

    @overrides
    def vre_free(self):
        assert self.setup_called is True and self.model is not None, (self.setup_called, self.model is not None)
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
        self.model = None
        self.setup_called = False

    def _fetch_weights(self) -> Path:
        """weights/<variant>.pt: on disk, else from VRE's gitlab, else from huggingface (same state_dict)"""
        path = DinoV2.get_weights_paths(self.variant)[0]
        try:
            return fetch_weights([path])[0]
        except URLError as e: # HTTPError (e.g. 404: not on gitlab) is a URLError
            url = HF_URL.format(variant=self.variant, revision=HF_REVISIONS[self.variant])
            logger.info(f"'{path.name}' not on VRE's gitlab ({e}). Falling back to huggingface: '{url}'")
            return fetch(url, path)

    def _preprocess(self, frames: np.ndarray) -> tr.Tensor:
        """uint8 (B, H, W, 3) -> ImageNet-normalized float (B, 3, *input_size)"""
        assert frames.ndim == 4 and frames.shape[-1] == 3 and frames.dtype == np.uint8, (frames.shape, frames.dtype)
        x = tr.from_numpy(frames).to(self.device).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False, antialias=True)
        mean = tr.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = tr.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _pool(self, tokens: tr.Tensor) -> tr.Tensor:
        """(B, 1 + N, D) last_hidden_state -> (B, n_channels), not yet L2-normalized"""
        if self.pool == "cls":
            return tokens[:, 0]
        if self.pool == "gem":
            return tokens[:, 1:].clamp(min=1e-6).pow(self.gem_p).mean(dim=1).pow(1.0 / self.gem_p)
        if self.pool == "mean":
            return tokens[:, 1:].mean(dim=1)
        raise ValueError(self.pool)

def _pca_rgb(grid: np.ndarray) -> np.ndarray:
    """
    (h, w, D) tokens -> (h, w, 3) uint8: the top-3 principal components over the h*w tokens, each min-max scaled.
    Each component's sign is fixed (its largest |loading| is positive), so the same grid always gives the same colors.
    The basis is per frame: colors are not comparable across frames.
    """
    h, w, d = grid.shape
    x = grid.reshape(h * w, d).astype(np.float64) # (h*w, d), d = number of dinov2 dims
    x = x - x.mean(axis=0) # (h*w, d)
    _, _, vt = np.linalg.svd(x, full_matrices=False) # (d, h*w)
    vt = vt[0:3] # fewer than 3 if h*w or D < 3, (3, h*w)
    signs = np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)]) # (3, )
    proj = np.zeros((h * w, 3))
    # multiply with signs so the final component per pixel is always positive I think?
    proj[:, 0:len(vt)] = x @ (vt * np.where(signs == 0, 1, signs)[:, None]).T # (h*w, d) @ (3, h*w)^T = (h*w, 3)
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    return np.round((proj - lo) / np.maximum(hi - lo, 1e-8) * 255).astype(np.uint8).reshape(h, w, 3)
