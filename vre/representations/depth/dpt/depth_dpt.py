"""DPT Depth Estimation representation"""
import numpy as np
import torch as tr
import torch.nn.functional as F
from overrides import overrides

from vre.utils import VREVideo, fetch_weights, colorize_depth, vre_load_weights, MemoryData
from vre.representations import (
    Representation, ReprOut, LearnedRepresentationMixin, ComputeRepresentationMixin, NpIORepresentation)
from vre.representations.depth.dpt.dpt_impl import DPTDepthModel, get_size

class DepthDpt(Representation, LearnedRepresentationMixin, ComputeRepresentationMixin, NpIORepresentation):
    """DPT Depth Estimation representation"""
    def __init__(self, **kwargs):
        Representation.__init__(self, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        self.net_w, self.net_h = 384, 384
        self.multiple_of = 32
        tr.manual_seed(42)
        self.model: DPTDepthModel | None = None

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        tr_frames = self._preprocess(video[ixs])
        with tr.no_grad():
            predictions = self.model(tr_frames)
        res = self._postprocess(predictions)
        self.data = ReprOut(frames=video[ixs], output=MemoryData(res), key=ixs)

    @overrides
    def make_images(self) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        return (colorize_depth(self.data.output, min_depth=0, max_depth=1) * 255).astype(np.uint8)

    @overrides
    def vre_setup(self, load_weights: bool = True):
        assert self.setup_called is False
        self.model = DPTDepthModel(backbone="vitl16_384", non_negative=True).to("cpu")
        if load_weights:
            weights_file = fetch_weights(__file__) / "depth_dpt_midas.pth"
            self.model.load_state_dict(vre_load_weights(weights_file))
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

    def _preprocess(self, x: np.ndarray) -> tr.Tensor:
        tr_frames = tr.from_numpy(x).to(self.device)
        tr_frames_perm = tr_frames.permute(0, 3, 1, 2).float() / 255
        curr_h, curr_w = tr_frames.shape[1], tr_frames.shape[2]
        h, w = get_size(self.net_h, self.net_w, curr_h, curr_w, multiple_of=self.multiple_of)
        tr_frames_resized = F.interpolate(tr_frames_perm, size=(h, w), mode="bicubic")
        tr_frames_norm = (tr_frames_resized - 0.5) / 0.5
        return tr_frames_norm

    def _postprocess(self, y: tr.Tensor) -> np.ndarray:
        return (1 / y).clip(0, 1).cpu().numpy()
