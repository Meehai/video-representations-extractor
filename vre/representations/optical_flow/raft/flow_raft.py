import os
import numpy as np
import pims
import torch as tr
from torch.nn import functional as F
import flow_vis
import gdown
from overrides import overrides
from pathlib import Path

from .utils import InputPadder
from .raft import RAFT
from ....representation import Representation, RepresentationOutput
from ....logger import logger

class FlowRaft(Representation):
    def __init__(self, video: pims.Video, name: str, dependencies: list[Representation],
                 inference_width: int, inference_height: int, iters: int,
                 small: bool, mixed_precision: bool, device: str):
        self.small = small
        self.mixed_precision = mixed_precision
        self.iters = iters
        self.device = device

        self.model = None
        assert tr.cuda.is_available() or self.device == "cpu", "CUDA not available"
        super().__init__(video, name, dependencies)
        self._setup()
        self.inference_width = inference_width
        self.inference_height = inference_height

        # Pointless to upsample with bilinear, it's better we fix the video input.
        assert self.video.shape[1] >= self.inference_height and self.video.shape[2] >= self.inference_width, \
            f"{self.video.shape} vs {self.inference_height}x{self.inference_width}"

    def _setup(self):
        weights_dir = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/raft")
        weights_dir.mkdir(exist_ok=True, parents=True)

        # original files
        raft_things_url = "https://drive.google.com/u/0/uc?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM"

        raft_things_path = weights_dir / "raft-things.pkl"
        if not raft_things_path.exists():
            logger.debug("Downloading weights for RAFT")
            gdown.download(raft_things_url, str(raft_things_path))

        model = RAFT(self)
        def convert(data: dict[str, tr.Tensor]) -> dict[str, tr.Tensor]:
            return {k.replace("module.", ""): v for k, v in data.items()}
        model.load_state_dict(convert(tr.load(raft_things_path, map_location="cpu")))
        model.eval()

        self.model = model.to(self.device)

    def _preprocess(self, frames: np.ndarray) -> tr.Tensor:
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
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        # [0 : 1] => [-1 : 1]
        x = x * 2 - 1
        y = np.array([flow_vis.flow_to_color(_x) for _x in x])
        return y
