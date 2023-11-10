"""FlowRaft representation"""
import os
import numpy as np
import pims
import torch as tr
from torch.nn import functional as F
import flow_vis
from overrides import overrides
from pathlib import Path

from .utils import InputPadder
from .raft import RAFT
from ....representation import Representation, RepresentationOutput
from ....utils import gdown_mkdir, image_resize_batch

class FlowRaft(Representation):
    """FlowRaft representation"""
    def __init__(self, video: pims.Video, name: str, dependencies: list[Representation], inference_height: int,
                 inference_width: int, iters: int, small: bool, mixed_precision: bool):
        self.small = small
        self.mixed_precision = mixed_precision
        self.iters = iters
        self.device = "cpu"

        tr.manual_seed(42)
        self.model = RAFT(self).to("cpu")
        super().__init__(video, name, dependencies)
        self.inference_width = inference_width
        self.inference_height = inference_height

        # Pointless to upsample with bilinear, it's better we fix the video input.
        assert self.video.shape[1] >= 128 and self.video.shape[2] >= 128, \
            f"This flow doesn't work with small videos. At least 128x128 is required, but got {self.video.shape}"
        assert self.inference_height >= 128 and self.inference_width >= 128, f"This flow doesn't work with small " \
            f"videos. At least 128x128 is required, but got {self.inference_height}x{self.inference_width}"
        assert self.video.shape[1] >= self.inference_height and self.video.shape[2] >= self.inference_width, \
            f"{self.video.shape} vs {self.inference_height}x{self.inference_width}"

    @overrides(check_signature=False)
    def vre_setup(self, device: str):
        weights_dir = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/raft")
        weights_dir.mkdir(exist_ok=True, parents=True)

        # original files
        raft_things_url = "https://drive.google.com/u/0/uc?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM"

        raft_things_path = weights_dir / "raft-things.pkl"
        if not raft_things_path.exists():
            gdown_mkdir(raft_things_url, raft_things_path)

        def convert(data: dict[str, tr.Tensor]) -> dict[str, tr.Tensor]:
            return {k.replace("module.", ""): v for k, v in data.items()}
        self.model.load_state_dict(convert(tr.load(raft_things_path, map_location="cpu")))
        self.model = self.model.eval().to(self.device)

    def _preprocess(self, frames: np.ndarray) -> (tr.Tensor, tr.Tensor):
        orig_height, orig_width = self.video.shape[1:3]
        tr_frames = tr.from_numpy(frames).to(self.device)
        tr_frames = tr_frames.permute(0, 3, 1, 2).float()

        frames_rsz = F.interpolate(tr_frames, (self.inference_height, self.inference_width), mode="bilinear")
        padder = InputPadder((len(frames), 3, self.inference_height, self.inference_width))
        frames_padded = padder.pad(frames_rsz)

        source, dest = frames_padded[:-1], frames_padded[1:]
        return source, dest

    def _postporcess(self, predictions: tr.Tensor) -> np.ndarray:
        padder = InputPadder((len(predictions), 3, self.inference_height, self.inference_width))
        flow_unpad = padder.unpad(predictions).cpu().numpy()
        flow_perm = flow_unpad.transpose(0, 2, 3, 1)
        flow_unpad_norm = flow_perm / (self.inference_height, self.inference_width)
        flow_unpad_norm = (flow_unpad_norm + 1) / 2
        flow_unpad_norm = flow_unpad_norm.astype(np.float32)
        return flow_unpad_norm

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        ts = [*list(range(t.start, t.stop)), min(t.stop + 1, len(self.video) - 1)]
        frames = np.array(self.video[ts])
        source, dest = self._preprocess(frames)
        with tr.no_grad():
            _, predictions = self.model(source, dest, iters=self.iters, test_mode=True)
        flow = self._postporcess(predictions)
        return flow

    @overrides
    def make_images(self, t: slice, x: np.ndarray, extra: dict | None) -> np.ndarray:
        # [0 : 1] => [-1 : 1]
        x = x * 2 - 1
        x_rsz = image_resize_batch(x, height=self.video.frame_shape[0], width=self.video.frame_shape[1])
        y = np.array([flow_vis.flow_to_color(_pred) for _pred in x_rsz])
        return y
