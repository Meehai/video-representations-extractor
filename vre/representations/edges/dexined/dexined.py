"""Dexined representation."""
import numpy as np
import torch as tr
from overrides import overrides

from vre.logger import vre_logger as logger
from vre.utils import image_resize_batch, fetch_weights, vre_load_weights, VREVideo, MemoryData
from vre.representations import (
    Representation, ReprOut, LearnedRepresentationMixin, ComputeRepresentationMixin, NpIORepresentation)
from vre.representations.edges.dexined.model_dexined import DexiNed as Model

class DexiNed(Representation, LearnedRepresentationMixin, ComputeRepresentationMixin, NpIORepresentation):
    """Dexined representation."""
    def __init__(self, **kwargs):
        Representation.__init__(self, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        self.model: Model | None = None
        self.inference_height, self.inference_width = 512, 512 # fixed for this model

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        tr_frames = self._preprocess(video[ixs], self.inference_height, self.inference_width)
        with tr.no_grad():
            y = self.model.forward(tr_frames)
        outs = self._postprocess(y)
        self.data = ReprOut(frames=video[ixs], output=MemoryData(outs), key=ixs)

    @overrides
    def make_images(self) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        x = np.repeat(np.expand_dims(self.data.output, axis=-1), 3, axis=-1)
        return (x * 255).astype(np.uint8)

    @overrides
    def vre_setup(self, load_weights: bool = True):
        assert self.setup_called is False
        self.model = Model().eval()
        if load_weights:
            self.model.load_state_dict(vre_load_weights(fetch_weights(__file__) / "dexined.pth"))
        self.model = self.model.to(self.device)
        self.setup_called = True

    @overrides
    def vre_free(self):
        assert self.setup_called is True and self.model is not None, (self.setup_called, self.model is not None)
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
        self.model = None
        self.setup_called = False

    def _preprocess(self, images: np.ndarray, height: int, width: int) -> tr.Tensor:
        assert len(images.shape) == 4, images.shape
        logger.debug2(f"Original shape: {images.shape}")
        images_resize = image_resize_batch(images, height, width)
        mean_pixel_values = [103.939, 116.779, 123.68]
        images_norm = images_resize.astype(np.float32) - mean_pixel_values
        # N, H, W, C -> N, C, H, W
        images_norm = images_norm.transpose((0, 3, 1, 2))
        tr_images_norm = tr.from_numpy(images_norm).to(self.device).float()
        return tr_images_norm

    def _postprocess(self, y: list[tr.Tensor]) -> np.ndarray:
        # y :: (B, T, H, W)
        y_cat = tr.cat(y, dim=1)
        assert len(y_cat.shape) == 4, y_cat.shape
        y_s = y_cat.sigmoid()
        A = y_s.min(dim=-1, keepdims=True)[0].min(dim=-2, keepdims=True)[0]
        B = y_s.max(dim=-1, keepdims=True)[0].max(dim=-2, keepdims=True)[0]
        y_s_normed: tr.Tensor = 1 - (y_s - A) * (B - A)
        y_s_final = y_s_normed.mean(dim=1).cpu().numpy().astype(np.float32)
        return y_s_final
