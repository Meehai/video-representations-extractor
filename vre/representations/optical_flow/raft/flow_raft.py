"""FlowRaft representation"""
from overrides import overrides
import numpy as np
import torch as tr
from torch.nn import functional as F
import flow_vis

from .raft_impl.utils import InputPadder
from .raft_impl.raft import RAFT
from ....representation import Representation, RepresentationOutput
from ....utils import image_resize_batch, fetch_weights, load_weights
from ....logger import vre_logger as logger

class FlowRaft(Representation):
    """FlowRaft representation"""
    def __init__(self, inference_height: int, inference_width: int, iters: int, small: bool, **kwargs):
        assert inference_height >= 128 and inference_width >= 128, f"This flow doesn't work with small " \
            f"videos. At least 128x128 is required, but got {inference_height}x{inference_width}"
        self.mixed_precision = False
        self.small = small
        self.iters = iters

        tr.manual_seed(42)
        self.model = RAFT(self).to("cpu")
        super().__init__(**kwargs)
        self.inference_width = inference_width
        self.inference_height = inference_height

    @overrides
    def vre_setup(self):
        assert self.video.frame_shape[0] >= 128 and self.video.frame_shape[1] >= 128, \
            f"This flow doesn't work with small videos. At least 128x128 is required, but got {self.video.shape}"
        assert self.video.frame_shape[0] >= self.inference_height \
            and self.video.frame_shape[1] >= self.inference_width, \
            f"{self.video.frame_shape} vs {self.inference_height}x{self.inference_width}"

        def convert(data: dict[str, tr.Tensor]) -> dict[str, tr.Tensor]:
            logger.warning("REMOVE THIS WHEN THERE'S TIME")
            return {k.replace("module.", ""): v for k, v in data.items()}
        raft_things_path = fetch_weights(__file__) / "raft-things.pkl"
        self.model.load_state_dict(convert(load_weights(raft_things_path)))
        self.model = self.model.eval().to(self.device)

    @overrides
    def vre_dep_data(self, ix: slice) -> dict[str, RepresentationOutput]:
        right_frames = np.array(self.video[ix.start + 1: min(ix.stop + 1, len(self.video))])
        if ix.stop + 1 > len(self.video):
            right_frames = np.concatenate([right_frames, np.array([self.video[-1]])], axis=0)
        return {"right_frames": right_frames}

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, RepresentationOutput] | None = None) -> RepresentationOutput:
        right_frames = dep_data["right_frames"]
        assert frames.shape == right_frames.shape, (frames.shape, right_frames.shape)
        assert frames.shape[1] >= self.inference_height and frames.shape[2] >= self.inference_width, \
            f"{frames.shape} vs {self.inference_height}x{self.inference_width}. Must be at least 128x128 usually."
        source, dest = self._preprocess(frames), self._preprocess(right_frames)
        with tr.no_grad():
            _, predictions = self.model(source, dest, iters=self.iters, test_mode=True)
        flow = self._postporcess(predictions)
        return RepresentationOutput(output=flow)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        y = np.array([flow_vis.flow_to_color(_pred) for _pred in repr_data.output])
        return y

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return RepresentationOutput(output=image_resize_batch(repr_data.output, *new_size))

    def _preprocess(self, frames: np.ndarray) -> tuple[tr.Tensor, tr.Tensor]:
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
        flow_unpad_norm = flow_unpad_norm.astype(np.float16)
        return flow_unpad_norm

    def vre_free(self):
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
