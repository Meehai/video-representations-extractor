import os
import numpy as np
import torch as tr
import torch.nn.functional as F
from overrides import overrides
from pathlib import Path
from matplotlib.cm import hot

from .dpt_depth import DPTDepthModel
from ....representation import Representation, RepresentationOutput
from ....utils import gdown_mkdir, image_resize_batch

def constrain_to_multiple_of(x, multiple_of: int, min_val=0, max_val=None):
    y = (np.round(x / multiple_of) * multiple_of).astype(int)
    if max_val is not None and y > max_val:
        y = (np.floor(x / multiple_of) * multiple_of).astype(int)
    if y < min_val:
        y = (np.ceil(x / multiple_of) * multiple_of).astype(int)
    return y

def get_size(__height, __width, height, width, multiple_of):
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
    new_height = constrain_to_multiple_of(scale_height * height, multiple_of)
    new_width = constrain_to_multiple_of(scale_width * width, multiple_of)

    return new_height, new_width


class DepthDpt(Representation):
    def __init__(self, **kwargs):
        self.model = None
        super().__init__(**kwargs)
        # VRE setup stuff
        self.net_w, self.net_h = 384, 384
        self.multiple_of = 32
        self.device: str = "cpu"
        self._setup()

    def _preprocess(self, x: np.ndarray) -> tr.Tensor:
        tr_frames = tr.from_numpy(x).to(self.device)
        tr_frames_perm = tr_frames.permute(0, 3, 1, 2).float() / 255
        curr_h, curr_w = tr_frames.shape[1], tr_frames.shape[2]
        h, w = get_size(self.net_h, self.net_w, curr_h, curr_w, multiple_of=self.multiple_of)
        tr_frames_resized = F.interpolate(tr_frames_perm, size=(int(h), int(w)), mode="bicubic")
        tr_frames_norm = (tr_frames_resized - 0.5) / 0.5
        return tr_frames_norm

    def _postprocess(self, y: tr.Tensor) -> np.ndarray:
        return (1 / y).clip(0, 1).cpu().numpy()

    def _setup(self):
        tr.manual_seed(42)
        self.model = DPTDepthModel(backbone="vitl16_384", non_negative=True).to("cpu")

    @overrides(check_signature=False)
    def vre_setup(self, device: str):
        assert tr.cuda.is_available() or device == "cpu", "CUDA not available"
        # our backup
        weights_file = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/depth_dpt_midas.pth").absolute()
        url_weights = "https://drive.google.com/u/0/uc?id=15JbN2YSkZFSaSV2CGkU1kVSxCBrNtyhD"

        if not weights_file.exists():
            gdown_mkdir(url_weights, weights_file)

        self.device = device
        self.model.load_state_dict(tr.load(weights_file, map_location="cpu"))
        self.model = self.model.eval().to(self.device)

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        frames = np.array(self.video[t])
        tr_frames = self._preprocess(frames)
        with tr.no_grad():
            predictions = self.model(tr_frames)
        res = self._postprocess(predictions)
        return res

    @overrides
    def make_images(self, t: slice, x: np.ndarray, extra: dict | None) -> np.ndarray:
        x_rsz = image_resize_batch(x, height=self.video.frame_shape[0], width=self.video.frame_shape[1])
        y = hot(x_rsz)[..., 0:3]
        y = np.uint8(y * 255)
        return y
