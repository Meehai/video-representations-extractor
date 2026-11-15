"""DINOv2 global image embeddings. Weights are transformers' facebook/dinov2-* state_dicts (see README.md)."""
from pathlib import Path
import numpy as np
import torch as tr
from torch.nn import functional as F
from overrides import overrides

from vre_video import VREVideo
from vre.utils import MemoryData
from vre import ReprOut
from vre.representations.mixins import LearnedRepresentationMixin
from vre_repository.utils import fetch_weights
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
POOLS = ("cls", "gem", "cat")

class DinoV2(EmbeddingRepresentation, LearnedRepresentationMixin):
    """
    One L2-normalized DINOv2 vector per frame.
    - variant: one of VARIANTS ('testing' = tiny random model, no weights).
    - pool: 'cls' = the CLS token, 'gem' = GeM (p=3) over the patch tokens, 'cat' = both, each L2-normalized (2*D).
    - long_side: frames are resized (aspect kept) so the long side is this, each side rounded to a multiple of 14.
    On cuda the model runs under float16 autocast.
    """
    def __init__(self, variant: str, pool: str = "cat", long_side: int = 672, **kwargs):
        EmbeddingRepresentation.__init__(self, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        if variant not in VARIANTS:
            raise ValueError(f"{variant=} not in {list(VARIANTS)}")
        if pool not in POOLS:
            raise ValueError(f"{pool=} not in {POOLS}")
        if long_side < 14:
            raise ValueError(f"{long_side=} must be at least the patch size (14)")
        self.variant = variant
        self.pool = pool
        self.long_side = long_side
        self.model: Dinov2Model | None = None

    @property
    @overrides
    def n_channels(self) -> int:
        hidden_size = VARIANTS[self.variant][0]
        return 2 * hidden_size if self.pool == "cat" else hidden_size

    @overrides
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        assert self.setup_called is True and self.model is not None, (self.setup_called, self.model is not None)
        frames = video[ixs]
        with tr.inference_mode():
            x = self._preprocess(frames)
            with tr.autocast(device_type="cuda", dtype=tr.float16, enabled=str(self.device).startswith("cuda")):
                tokens = self.model(x)
            embeddings = F.normalize(self._pool(tokens.float()), dim=-1).cpu().numpy()
        return ReprOut(frames=frames, output=MemoryData(embeddings), key=ixs)

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
            path = fetch_weights(DinoV2.get_weights_paths(self.variant))[0]
            self.model.load_state_dict(tr.load(path, map_location="cpu"))
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

    def _preprocess(self, frames: np.ndarray) -> tr.Tensor:
        """uint8 (B, H, W, 3) -> ImageNet-normalized float (B, 3, H', W'), long side ~long_side, sides multiple of 14"""
        assert frames.ndim == 4 and frames.shape[-1] == 3 and frames.dtype == np.uint8, (frames.shape, frames.dtype)
        h, w = frames.shape[1:3]
        patch = self.model.patch_size
        scale = self.long_side / max(h, w)
        new_h = max(patch, round(h * scale / patch) * patch)
        new_w = max(patch, round(w * scale / patch) * patch)
        x = tr.from_numpy(frames).to(self.device).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        mean = tr.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = tr.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _pool(self, tokens: tr.Tensor) -> tr.Tensor:
        """(B, 1 + N, D) last_hidden_state -> (B, n_channels), not yet L2-normalized"""
        cls = tokens[:, 0]
        gem = tokens[:, 1:].clamp(min=1e-6).pow(3.0).mean(dim=1).pow(1.0 / 3.0)
        if self.pool == "cls":
            return cls
        if self.pool == "gem":
            return gem
        return tr.cat([F.normalize(cls, dim=-1), F.normalize(gem, dim=-1)], dim=-1)
