"""FlowRife representation"""
from pathlib import Path
import os
import pims
import numpy as np
import torch as tr
import torch.nn.functional as F
import flow_vis
from overrides import overrides

from ....utils import gdown_mkdir
from ....representation import Representation, RepresentationOutput
from .RIFE_HDv2 import Model

"""FlowRife representation"""
class FlowRife(Representation):
    def __init__(self, video: pims.Video, name: str, dependencies: list[Representation],
                 compute_backward_flow: bool, uhd: bool):
        self.model: Model = Model().eval().to("cpu")
        self.uhd = uhd
        self.no_backward_flow = True if compute_backward_flow is None else not compute_backward_flow
        self.device = "cpu"
        assert tr.cuda.is_available() or self.device == "cpu", "CUDA not available"
        super().__init__(video, name, dependencies)

    @overrides(check_signature=False)
    def vre_setup(self, device: str):
        self.device = device
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
            gdown_mkdir(contextnet_url, contextnet_path)

        flownet_path = weights_dir / "flownet.pkl"
        if not flownet_path.exists():
            gdown_mkdir(flownet_url, flownet_path)

        unet_path = weights_dir / "unet.pkl"
        if not unet_path.exists():
            gdown_mkdir(unet_url, unet_path)

        self.model.load_model(weights_dir)
        self.model = self.model.eval().to(self.device)

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        # add t+1 to have one more frame in targets. If it's the last frame, we add the same frame.
        ts = [*list(range(t.start, t.stop)), min(t.stop + 1, len(self.video) - 1)]
        frames = np.array(self.video[ts])

        sources = frames[0: -1].transpose(0, 3, 1, 2)
        targets = frames[1:].transpose(0, 3, 1, 2)

        x_s, x_t, padding = self._preprocess(sources, targets)
        with tr.no_grad():
            prediction = self.model.inference(x_s, x_t, self.uhd, self.no_backward_flow)
        flow = self._postprocess(prediction, padding)
        return flow

    @overrides
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        # [0 : 1] => [-1 : 1]
        x = x * 2 - 1
        y = np.array([flow_vis.flow_to_color(_pred) for _pred in x])
        return y

    def _preprocess(self, sources: np.ndarray, targets: np.ndarray) -> (tr.Tensor, tr.Tensor, tuple):
        # Convert, preprocess & pad
        I0 = tr.from_numpy(sources).to(self.device, non_blocking=True).float() / 255.0
        I1 = tr.from_numpy(targets).to(self.device, non_blocking=True).float() / 255.0
        n, c, h, w = I0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        I0 = F.pad(I0, padding)
        I1 = F.pad(I1, padding)
        return I0, I1, padding

    def _postprocess(self, prediction: tr.Tensor, padding: tuple) -> np.ndarray:
        flow = prediction.cpu().numpy().transpose(0, 2, 3, 1)
        returned_shape = flow.shape[1:3]
        # Remove the padding to keep original shape
        half_ph, half_pw = padding[3] // 2, padding[1] // 2
        flow = flow[:, 0: returned_shape[0] - half_ph, 0: returned_shape[1] - half_pw]
        # [-px : px] => [-1 : 1]
        flow /= returned_shape
        # [-1 : 1] => [0 : 1]
        flow = (flow + 1) / 2
        return flow
