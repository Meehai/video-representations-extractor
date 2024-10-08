"""DPT Depth Estimation representation"""
import numpy as np
import torch as tr
import torch.nn.functional as F
from overrides import overrides

from .dpt_impl.dpt_depth import DPTDepthModel
from ....representation import Representation, RepresentationOutput
from ....utils import image_resize_batch, fetch_weights, colorize_depth_maps, load_weights

def _constrain_to_multiple_of(x, multiple_of: int, min_val=0, max_val=None) -> int:
    y = (np.round(x / multiple_of) * multiple_of).astype(int)
    if max_val is not None and y > max_val:
        y = (np.floor(x / multiple_of) * multiple_of).astype(int)
    if y < min_val:
        y = (np.ceil(x / multiple_of) * multiple_of).astype(int)
    return int(y)

def _get_size(__height, __width, height, width, multiple_of) -> tuple[int, int]:
    # determine new height and width
    scale_height = __height / height
    scale_width = __width / width
    # keep aspect ratio
    if abs(1 - scale_width) < abs(1 - scale_height):
        # fit width
        scale_height = scale_width
    else:
        # fit height
        scale_width = scale_height
    new_height = _constrain_to_multiple_of(scale_height * height, multiple_of)
    new_width = _constrain_to_multiple_of(scale_width * width, multiple_of)

    return new_height, new_width


class DepthDpt(Representation):
    """DPT Depth Estimation representation"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # VRE setup stuff
        self.net_w, self.net_h = 384, 384
        self.multiple_of = 32
        tr.manual_seed(42)
        self.model = DPTDepthModel(backbone="vitl16_384", non_negative=True).to("cpu")

    @overrides
    def vre_setup(self):
        weights_file = fetch_weights(__file__) / "depth_dpt_midas.pth"
        self.model.load_state_dict(load_weights(weights_file))
        self.model = self.model.eval().to(self.device)

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, RepresentationOutput] | None = None) -> RepresentationOutput:
        tr_frames = self._preprocess(frames)
        with tr.no_grad():
            predictions = self.model(tr_frames)
        res = self._postprocess(predictions)
        return RepresentationOutput(output=res)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        return (colorize_depth_maps(repr_data.output, min_depth=0, max_depth=1) * 255).astype(np.uint8)

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return RepresentationOutput(output=image_resize_batch(repr_data.output, *new_size))

    @overrides
    def vre_free(self):
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()

    def _preprocess(self, x: np.ndarray) -> tr.Tensor:
        tr_frames = tr.from_numpy(x).to(self.device)
        tr_frames_perm = tr_frames.permute(0, 3, 1, 2).float() / 255
        curr_h, curr_w = tr_frames.shape[1], tr_frames.shape[2]
        h, w = _get_size(self.net_h, self.net_w, curr_h, curr_w, multiple_of=self.multiple_of)
        tr_frames_resized = F.interpolate(tr_frames_perm, size=(h, w), mode="bicubic")
        tr_frames_norm = (tr_frames_resized - 0.5) / 0.5
        return tr_frames_norm

    def _postprocess(self, y: tr.Tensor) -> np.ndarray:
        return (1 / y).clip(0, 1).cpu().numpy()
