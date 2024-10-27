"""DPT Depth Estimation representation"""
import numpy as np
import torch as tr
import torch.nn.functional as F
from overrides import overrides

from vre.utils import image_resize_batch, fetch_weights, colorize_depth, vre_load_weights
from vre.representations import Representation, ReprOut, LearnedRepresentationMixin, ComputeRepresentationMixin
from vre.representations.depth.dpt.dpt_impl import DPTDepthModel, get_size

class DepthDpt(Representation, LearnedRepresentationMixin, ComputeRepresentationMixin):
    """DPT Depth Estimation representation"""
    def __init__(self, *args, **kwargs):
        Representation.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.net_w, self.net_h = 384, 384
        self.multiple_of = 32
        tr.manual_seed(42)
        self.model: DPTDepthModel | None = None

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
    def make(self, frames: np.ndarray, dep_data: dict[str, ReprOut] | None = None) -> ReprOut:
        tr_frames = self._preprocess(frames)
        with tr.no_grad():
            predictions = self.model(tr_frames)
        res = self._postprocess(predictions)
        return ReprOut(output=res)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: ReprOut) -> np.ndarray:
        return (colorize_depth(repr_data.output, min_depth=0, max_depth=1) * 255).astype(np.uint8)

    @overrides
    def size(self, repr_data: ReprOut) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: ReprOut, new_size: tuple[int, int]) -> ReprOut:
        return ReprOut(output=image_resize_batch(repr_data.output, *new_size))

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
