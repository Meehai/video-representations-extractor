"""DPT Depth Estimation representation"""
import numpy as np
import torch as tr
import torch.nn.functional as F
from overrides import overrides
from matplotlib.cm import hot # pylint: disable=no-name-in-module

from .dpt_impl.dpt_depth import DPTDepthModel
from ....representation import Representation, RepresentationOutput
from ....utils import gdown_mkdir, image_resize_batch, VREVideo, get_weights_dir
from ....logger import logger

def _constrain_to_multiple_of(x, multiple_of: int, min_val=0, max_val=None) -> int:
    y = (np.round(x / multiple_of) * multiple_of).astype(int)
    if max_val is not None and y > max_val:
        y = (np.floor(x / multiple_of) * multiple_of).astype(int)
    if y < min_val:
        y = (np.ceil(x / multiple_of) * multiple_of).astype(int)
    return int(y)

def _get_size(__height, __width, height, width, multiple_of) -> (int, int):
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
        self.device: str = "cpu"
        tr.manual_seed(42)
        self.model = DPTDepthModel(backbone="vitl16_384", non_negative=True).to("cpu")

    # pylint: disable=arguments-differ
    @overrides(check_signature=False)
    def vre_setup(self, video: VREVideo, device: str):
        assert tr.cuda.is_available() or device == "cpu", "CUDA not available"
        # our backup
        weights_file = get_weights_dir() / "depth_dpt_midas.pth"
        url_weights = "https://drive.google.com/u/0/uc?id=15JbN2YSkZFSaSV2CGkU1kVSxCBrNtyhD"

        if not weights_file.exists():
            gdown_mkdir(url_weights, weights_file)

        self.device = device
        logger.info(f"Loading weights from '{weights_file}'")
        weights_data = tr.load(weights_file, map_location="cpu")
        self.model.load_state_dict(weights_data)
        self.model = self.model.eval().to(self.device)

    @overrides
    def make(self, frames: np.ndarray) -> RepresentationOutput:
        tr_frames = self._preprocess(frames)
        with tr.no_grad():
            predictions = self.model(tr_frames)
        res = self._postprocess(predictions)
        return res

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        y = hot(repr_data)[..., 0:3]
        y = np.uint8(y * 255)
        return y

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return image_resize_batch(repr_data, *new_size)

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
