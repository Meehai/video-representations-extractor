"""FlowRife representation"""
import numpy as np
import torch as tr
import torch.nn.functional as F
from overrides import overrides

from vre_video import VREVideo
from vre.utils import MemoryData
from vre.representations import ReprOut, LearnedRepresentationMixin
from vre_repository.optical_flow import OpticalFlowRepresentation
from vre_repository.weights_repository import fetch_weights

from .rife_impl import Model

class FlowRife(OpticalFlowRepresentation, LearnedRepresentationMixin):
    """FlowRife representation"""
    def __init__(self, compute_backward_flow: bool, uhd: bool, flow_delta_frames: int = 1, **kwargs):
        OpticalFlowRepresentation.__init__(self, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        self.uhd = uhd
        self.flow_delta_frames = flow_delta_frames
        assert compute_backward_flow is False, "Not supported"
        self.no_backward_flow = True if compute_backward_flow is None else not compute_backward_flow
        self.model: Model | None = None

    @overrides
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        frames = video[ixs]
        right_frames = self.get_delta_frames(video, ixs)
        x_s, x_t, padding = self._preprocess(frames, right_frames)
        with tr.no_grad():
            prediction = self.model.inference(x_s, x_t, self.uhd, self.no_backward_flow)
        flow = self._postprocess(prediction, padding)
        return ReprOut(frames=video[ixs], output=MemoryData(flow), key=ixs)

    @staticmethod
    @overrides
    def weights_repository_links(**kwargs) -> list[str]:
        return [
            "optical_flow/rife/contextnet.ckpt",
            "optical_flow/rife/unet.ckpt",
            "optical_flow/rife/flownet.ckpt"
        ]

    @overrides
    def vre_setup(self, load_weights: bool = True):
        assert self.setup_called is False
        self.model: Model = Model().eval()
        if load_weights:
            paths = fetch_weights(FlowRife.weights_repository_links())
            self.model.load_model(paths[0].parent)
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
