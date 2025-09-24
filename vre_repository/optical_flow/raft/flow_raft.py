"""FlowRaft representation"""
from pathlib import Path
import numpy as np
import torch as tr
from torch.nn import functional as F
from overrides import overrides

from vre_video import VREVideo
from vre.utils import MemoryData
from vre.representations import ReprOut, LearnedRepresentationMixin
from vre_repository.optical_flow import OpticalFlowRepresentation
from vre_repository.weights_repository import fetch_weights

from .raft_impl import RAFT, InputPadder

# TODO: make inference_height/width a tuple inference_size
class FlowRaft(OpticalFlowRepresentation, LearnedRepresentationMixin):
    """FlowRaft representation"""
    def __init__(self, inference_height: int, inference_width: int, iters: int, small: bool,
                 seed: int | None = None, flow_delta_frames: int = 1, **kwargs):
        OpticalFlowRepresentation.__init__(self, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        assert inference_height >= 128 and inference_width >= 128, f"This flow doesn't work with small " \
            f"videos. At least 128x128 is required, but got {inference_height}x{inference_width}"
        self.mixed_precision = False
        self.small = small
        self.iters = iters
        self.seed = seed
        self.flow_delta_frames = flow_delta_frames

        self.model: RAFT | None = None
        self.inference_width = inference_width
        self.inference_height = inference_height

    @overrides
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        frames = video[ixs]
        right_frames = self.get_delta_frames(video, ixs)
        source, dest = self._preprocess(frames), self._preprocess(right_frames)
        tr.manual_seed(self.seed) if self.seed is not None else None
        with tr.no_grad():
            _, predictions = self.model(source, dest, iters=self.iters, test_mode=True)
        flow = self._postporcess(predictions)
        return ReprOut(frames=video[ixs], output=MemoryData(flow), key=ixs)

    @staticmethod
    @overrides
    def get_weights_paths(variant: str | None = None) -> list[str]:
        assert variant is None, variant
        return [Path(__file__).parent / "weights/raft-things.ckpt"]

    @overrides
    def vre_setup(self, load_weights: bool = True):
        assert self.setup_called is False
        tr.manual_seed(self.seed) if self.seed is not None else None
        self.model = RAFT(self).to("cpu")
        if load_weights:
            weights_path = fetch_weights(FlowRaft.get_weights_paths())[0]
            self.model.load_state_dict(tr.load(weights_path, map_location="cpu"))
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

    def _check_frames_resolution_requirements(self, frames: np.ndarray):
        assert frames.shape[1] >= 128 and frames.shape[2] >= 128, \
            f"This flow doesn't work with small videos. At least 128x128 is required, but got {frames.shape}"
        assert frames.shape[1] >= self.inference_height and frames.shape[2] >= self.inference_width, \
            f"{frames.shape} vs {self.inference_height}x{self.inference_width}"

    def _preprocess(self, frames: np.ndarray) -> tuple[tr.Tensor, tr.Tensor]:
        self._check_frames_resolution_requirements(frames)
        tr_frames = tr.from_numpy(frames).to(self.device)
        tr_frames = tr_frames.permute(0, 3, 1, 2).float() # (B, C, H, W)
        frames_rsz = F.interpolate(tr_frames, (self.inference_height, self.inference_width), mode="bilinear")
        padder = InputPadder((len(frames), 3, self.inference_height, self.inference_width))
        frames_padded = padder.pad(frames_rsz)
        return frames_padded

    def _postporcess(self, predictions: tr.Tensor) -> np.ndarray:
        padder = InputPadder((len(predictions), 3, self.inference_height, self.inference_width))
        flow_unpad = padder.unpad(predictions).cpu().numpy()
        flow_perm = flow_unpad.transpose(0, 2, 3, 1)
        flow_unpad_norm = flow_perm / (self.inference_height, self.inference_width) # [-1 : 1]
        return flow_unpad_norm.astype(np.float32)
