import os
import numpy as np
import pims
import torch as tr
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
                 inputWidth: int, inputHeight: int, device: str):
        self.model = None
        self.small = False
        self.mixed_precision = False
        self.device = device
        assert tr.cuda.is_available() or self.device == "cpu", "CUDA not available"
        super().__init__(video, name, dependencies)
        self._setup()
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight

        # Pointless to upsample with bilinear, it's better we fix the video input.
        assert (
            self.video.shape[1] >= self.inputHeight and self.video.shape[2] >= self.inputWidth
        ), f"{self.video.shape} vs {self.inputHeight}x{self.inputWidth}"

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

    @overrides
    def make(self, t: int) -> RepresentationOutput:
        t_target = t + 1 if t < len(self.video) - 1 else t
        return self.get(t, t_target)

    def get(self, t_source, t_target) -> np.ndarray:
        frame1 = self.video[t_source]
        frame2 = self.video[t_target]

        frame1 = cv2.resize(frame1, (self.inputWidth, self.inputHeight), interpolation=cv2.INTER_LINEAR)
        frame2 = cv2.resize(frame2, (self.inputWidth, self.inputHeight), interpolation=cv2.INTER_LINEAR)

        # Convert, preprocess & pad
        frame1 = tr.from_numpy(np.transpose(frame1, (2, 0, 1))).unsqueeze(0).float().to(self.device)
        frame2 = tr.from_numpy(np.transpose(frame2, (2, 0, 1))).unsqueeze(0).float().to(self.device)

        padder = InputPadder(frame1.shape)
        image1, image2 = padder.pad(frame1, frame2)

        with tr.no_grad():
            _, flow = self.model(image1, image2, iters=20, test_mode=True)

        # Convert, postprocess and remove pad
        flow = flow[0].cpu().numpy().transpose(1, 2, 0)
        returnedShape = flow.shape[0:2]
        # Remove the padding to keep original shape
        flow = padder.unpad(flow)
        # [-px : px] => [-1 : 1]
        flow /= returnedShape
        # [-1 : 1] => [0 : 1]
        flow = (flow + 1) / 2
        return flow

    @overrides
    def make_image(self, x: RepresentationOutput) -> np.ndarray:
        # [0 : 1] => [-1 : 1]
        x = x["data"] * 2 - 1
        y = flow_vis.flow_to_color(x)
        return y
