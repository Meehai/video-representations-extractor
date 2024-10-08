"""FlowRife representation"""
import numpy as np
import torch as tr
import torch.nn.functional as F
import flow_vis
from overrides import overrides

from ....utils import image_resize_batch, fetch_weights
from ....representation import Representation, RepresentationOutput
from .rife_impl.RIFE_HDv2 import Model

class FlowRife(Representation):
    """FlowRife representation"""
    def __init__(self, compute_backward_flow: bool, uhd: bool, **kwargs):
        tr.manual_seed(42)
        self.model: Model = Model().eval().to("cpu")
        self.uhd = uhd
        assert compute_backward_flow is False, "Not supported"
        self.no_backward_flow = True if compute_backward_flow is None else not compute_backward_flow
        super().__init__(**kwargs)

    @overrides
    def vre_setup(self):
        self.model.load_model(fetch_weights(__file__))
        self.model = self.model.eval().to(self.device)

    @overrides
    def vre_dep_data(self, ix: slice) -> dict[str, RepresentationOutput]:
        right_frames = np.array(self.video[ix.start + 1: min(ix.stop + 1, len(self.video))])
        if ix.stop + 1 > len(self.video):
            right_frames = np.concatenate([right_frames, np.array([self.video[-1]])], axis=0)
        return {"right_frames": RepresentationOutput(output=right_frames)}

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, RepresentationOutput] | None = None) -> RepresentationOutput:
        right_frames = dep_data["right_frames"].output
        x_s, x_t, padding = self._preprocess(frames, right_frames)
        with tr.no_grad():
            prediction = self.model.inference(x_s, x_t, self.uhd, self.no_backward_flow)
        flow = self._postprocess(prediction, padding)
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
        return flow.astype(np.float16)

    def vre_free(self):
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
