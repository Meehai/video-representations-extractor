import os
import numpy as np
import torch as tr
import torch.nn.functional as F
import flow_vis
import gdown
from pathlib import Path
from media_processing_lib.video import MPLVideo
from typing import Optional, List

try:
    from .RIFE_HDv2 import Model
    from ...representation import Representation, RepresentationOutput
    from ....logger import logger
except ImportError:
    from RIFE_HDv2 import Model
    from vre.representations.representation import Representation, RepresentationOutput
    from vre.logger import logger


device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")


class FlowRife(Representation):
    def __init__(
        self, video: MPLVideo, name: str, dependencies: List[Representation], computeBackwardFlow: Optional[bool] = None
    ):
        self.model = None
        self.UHD = False
        self.no_backward_flow = True if computeBackwardFlow is None else not computeBackwardFlow
        super().__init__(video, name, dependencies)

    def setup(self):
        weightsDir = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/rife").absolute()
        weightsDir.mkdir(exist_ok=True)

        # original files
        # urlWeights = "https://drive.google.com/u/0/uc?id=1wsQIhHZ3Eg4_AfCXItFKqqyDMB4NS0Yd"
        # our backup / dragos' better/sharper version
        contextNetUrl = "https://drive.google.com/u/0/uc?id=1x2_inKGBxjTYvdn58GyRnog0C7YdzE7-"
        flowNetUrl = "https://drive.google.com/u/0/uc?id=1aqR0ciMzKcD-N4bwkTK8go5FW4WAKoWc"
        uNetUrl = "https://drive.google.com/u/0/uc?id=1Fv27pNAbrmqQJolCFkD1Qm1RgKBRotME"

        contextNetPath = weightsDir / "contextnet.pkl"
        if not contextNetPath.exists():
            logger.debug("Downloading contextnet weights for RIFE")
            gdown.download(contextNetUrl, str(contextNetPath))

        flowNetPath = weightsDir / "flownet.pkl"
        if not flowNetPath.exists():
            logger.debug("Downloading flownet weights for RIFE")
            gdown.download(flowNetUrl, str(flowNetPath))

        uNetPath = weightsDir / "unet.pkl"
        if not uNetPath.exists():
            logger.debug("Downloading unet weights for RIFE")
            gdown.download(uNetUrl, str(uNetPath))

        if self.model is None:
            model = Model()
            model.load_model(weightsDir, -1)
            model.eval()
            model.device()
            self.model = model

    def make(self, t: int) -> RepresentationOutput:
        t_target = t + 1 if t < len(self.video) - 1 else t
        return self.get(t, t_target)

    def get(self, t_source, t_target) -> np.ndarray:
        frame1 = self.video[t_source]
        frame2 = self.video[t_target]

        # Convert, preprocess & pad
        I0 = tr.from_numpy(np.transpose(frame1, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.0
        I1 = tr.from_numpy(np.transpose(frame2, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.0
        n, c, h, w = I0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        I0 = F.pad(I0, padding)
        I1 = F.pad(I1, padding)

        with tr.no_grad():
            flow = self.model.inference(I0, I1, self.UHD, self.no_backward_flow)

        # Convert, postprocess and remove pad
        flow = flow[0].cpu().numpy().transpose(1, 2, 0)
        returnedShape = flow.shape[0:2]
        # Remove the padding to keep original shape
        halfPh, halfPw = (ph - h) // 2, (pw - w) // 2
        flow = flow[0 : returnedShape[0] - halfPh, 0 : returnedShape[1] - halfPw]
        # [-px : px] => [-1 : 1]
        flow /= returnedShape
        # [-1 : 1] => [0 : 1]
        flow = (flow + 1) / 2
        return flow

    def make_image(self, x: RepresentationOutput) -> np.ndarray:
        # [0 : 1] => [-1 : 1]
        x = x["data"] * 2 - 1
        y = flow_vis.flow_to_color(x)
        return y
