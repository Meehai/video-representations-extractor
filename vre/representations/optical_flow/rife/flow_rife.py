"""FlowRife representation"""
from pathlib import Path
import numpy as np
import torch as tr
import torch.nn.functional as F
from overrides import overrides

from vre.utils import image_resize_batch, fetch_weights, VREVideo, colorize_optical_flow
from vre.representations import Representation, ReprOut, LearnedRepresentationMixin, ComputeRepresentationMixin
from vre.representations.optical_flow.rife.rife_impl import Model

class FlowRife(Representation, LearnedRepresentationMixin, ComputeRepresentationMixin):
    """FlowRife representation"""
    def __init__(self, compute_backward_flow: bool, uhd: bool, flow_delta_frames: int = 1, **kwargs):
        Representation.__init__(self, **kwargs)
        super().__init__(**kwargs)
        tr.manual_seed(42)
        self.uhd = uhd
        self.flow_delta_frames = flow_delta_frames
        assert compute_backward_flow is False, "Not supported"
        self.no_backward_flow = True if compute_backward_flow is None else not compute_backward_flow
        self.model: Model | None = None

    @overrides
    def vre_setup(self, load_weights: bool = True):
        assert self.setup_called is False
        self.model: Model = Model().eval()
        if load_weights:
            self.model.load_model(fetch_weights(__file__))
        self.model = self.model.to(self.device)
        self.setup_called = True

    def vre_dep_data(self, video: VREVideo, ixs: slice | list[int], output_dir: Path | None) -> dict[str, ReprOut]:
        ixs: list[int] = list(range(ixs.start, ixs.stop)) if isinstance(ixs, slice) else ixs
        right_ixs = [min(ix + self.flow_delta_frames, len(video)) for ix in ixs]
        right_frames = np.array(video[right_ixs]) # godbless pims I hope it caches this properly
        return {"right_frames": ReprOut(output=right_frames)}

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, ReprOut] | None = None) -> ReprOut:
        right_frames = dep_data["right_frames"].output
        x_s, x_t, padding = self._preprocess(frames, right_frames)
        with tr.no_grad():
            prediction = self.model.inference(x_s, x_t, self.uhd, self.no_backward_flow)
        flow = self._postprocess(prediction, padding)
        return ReprOut(output=flow)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: ReprOut) -> np.ndarray:
        return np.array([colorize_optical_flow(_pred) for _pred in repr_data.output])

    @overrides
    def size(self, repr_data: ReprOut) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: ReprOut, new_size: tuple[int, int]) -> ReprOut:
        return ReprOut(output=image_resize_batch(repr_data.output, *new_size))

    def _preprocess(self, sources: np.ndarray, targets: np.ndarray) -> tuple[tr.Tensor, tr.Tensor, tuple]:
        # Convert, preprocess & pad
        sources = sources.transpose(0, 3, 1, 2)
        targets = targets.transpose(0, 3, 1, 2)
        tr_sources = tr.from_numpy(sources).to(self.device).float() / 255.0
        tr_targets = tr.from_numpy(targets).to(self.device).float() / 255.0
        h, w = tr_sources.shape[2:4]
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        tr_sources_padded = F.pad(tr_sources, padding) # pylint: disable=not-callable
        tr_target_padded = F.pad(tr_targets, padding) # pylint: disable=not-callable
        return tr_sources_padded, tr_target_padded, padding

    def _postprocess(self, prediction: tr.Tensor, padding: tuple) -> np.ndarray:
        flow = prediction.cpu().numpy().transpose(0, 2, 3, 1) # (B, H, W, C)
        returned_shape = flow.shape[1:3]
        # Remove the padding to keep original shape
        half_ph, half_pw = padding[3] // 2, padding[1] // 2
        flow = flow[:, 0: returned_shape[0] - half_ph, 0: returned_shape[1] - half_pw]
        flow = flow / returned_shape # [-px : px] => [-1 : 1]
        return flow.astype(np.float32)

    def vre_free(self):
        assert self.setup_called is True and self.model is not None, (self.setup_called, self.model is not None)
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
        self.model = None
        self.setup_called = False
