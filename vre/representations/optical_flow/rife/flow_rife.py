from pathlib import Path
import os
import pims
import numpy as np
import torch as tr
import torch.nn.functional as F
import flow_vis
import gdown

try:
    from .RIFE_HDv2 import Model
    from ....representation import Representation
    from ....logger import logger
except ImportError:
    from RIFE_HDv2 import Model
    from vre.representation import Representation
    from vre.logger import logger

class FlowRife(Representation):
    def __init__(self, video: pims.Video, name: str, dependencies: list[Representation],
                 computeBackwardFlow: bool, device: str):
        self.model = None
        self.UHD = False
        self.no_backward_flow = True if computeBackwardFlow is None else not computeBackwardFlow
        self.device = device
        assert tr.cuda.is_available() or self.device == "cpu", "CUDA not available"
        super().__init__(video, name, dependencies)
        self._setup()

    def _setup(self):
        weights_dir = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/rife").absolute()
        weights_dir.mkdir(exist_ok=True, parents=True)

        # original files
        # urlWeights = "https://drive.google.com/u/0/uc?id=1wsQIhHZ3Eg4_AfCXItFKqqyDMB4NS0Yd"
        # our backup / dragos' better/sharper version
        contextnet_url = "https://drive.google.com/u/0/uc?id=1x2_inKGBxjTYvdn58GyRnog0C7YdzE7-"
        flownet_url = "https://drive.google.com/u/0/uc?id=1aqR0ciMzKcD-N4bwkTK8go5FW4WAKoWc"
        unet_url = "https://drive.google.com/u/0/uc?id=1Fv27pNAbrmqQJolCFkD1Qm1RgKBRotME"

        contextnet_path = weights_dir / "contextnet.pkl"
        if not contextnet_path.exists():
            logger.debug("Downloading contextnet weights for RIFE")
            gdown.download(contextnet_url, str(contextnet_path))

        flownet_path = weights_dir / "flownet.pkl"
        if not flownet_path.exists():
            logger.debug("Downloading flownet weights for RIFE")
            gdown.download(flownet_url, str(flownet_path))

        unet_path = weights_dir / "unet.pkl"
        if not unet_path.exists():
            logger.debug("Downloading unet weights for RIFE")
            gdown.download(unet_url, str(unet_path))

        if self.model is None:
            model = Model()
            model.load_model(weights_dir)
            model.eval()
            self.model = model.to(self.device)

    def make(self, t: int) -> np.ndarray:
        t_target = t + 1 if t < len(self.video) - 1 else t
        return self.get(t, t_target)

    def get(self, t_source, t_target) -> np.ndarray:
        frame1 = self.video[t_source].transpose(2, 0, 1)[None]
        frame2 = self.video[t_target].transpose(2, 0, 1)[None]

        # Convert, preprocess & pad
        I0 = tr.from_numpy(frame1).to(self.device, non_blocking=True).float() / 255.0
        I1 = tr.from_numpy(frame2).to(self.device, non_blocking=True).float() / 255.0
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

    def make_image(self, x: np.ndarray) -> np.ndarray:
        # [0 : 1] => [-1 : 1]
        x = x["data"] * 2 - 1
        y = flow_vis.flow_to_color(x)
        return y
