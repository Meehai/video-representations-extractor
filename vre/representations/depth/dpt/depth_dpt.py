import os
import numpy as np
import torch as tr
import torch.nn.functional as F
import cv2
import gdown
from overrides import overrides
from torchvision.transforms import Compose
from pathlib import Path
from matplotlib.cm import hot

from .dpt_depth import DPTDepthModel
from .transforms import Resize, NormalizeImage, PrepareForNet
from ....representation import Representation
from ....logger import logger


def closest_fit(size, multiples):
    return [round(size[i] / multiples[i]) * multiples[i] for i in range(len(multiples))]


class DepthDpt(Representation):
    def __init__(self, device: str, **kwargs):
        self.model = None
        self.device = device
        assert tr.cuda.is_available() or self.device == "cpu", "CUDA not available"
        super().__init__(**kwargs)
        self._setup()
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

    def _setup(self):
        # our backup
        weights_file = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/depth_dpt_midas.pth").absolute()
        urlWeights = "https://drive.google.com/u/0/uc?id=15JbN2YSkZFSaSV2CGkU1kVSxCBrNtyhD"

        if not weights_file.exists():
            logger.debug(f"Downloading weights for dexined from {urlWeights}")
            gdown.download(urlWeights, str(weights_file))

        model = DPTDepthModel(backbone="vitl16_384", non_negative=True)
        model.load_state_dict(tr.load(weights_file, map_location="cpu"))
        model.eval()
        self.model = model.to(self.device)

    @overrides
    def make(self, t: slice) -> np.ndarray:
        raise NotImplementedError
        x = self.video[t]
        img_input = self.transform({"image": x / 255.0})["image"]
        # compute
        with tr.no_grad():
            sample = tr.from_numpy(img_input).unsqueeze(0).to(self.device)
            prediction = self.model.forward(sample).squeeze(dim=1)
            prediction = prediction.squeeze().cpu().numpy()
            prediction = 1 / prediction
            prediction = np.clip(prediction, 0, 1)
        return prediction

    @overrides
    def make_image(self, x: np.ndarray) -> np.ndarray:
        y = x["data"]
        y = hot(y)[..., 0:3]
        y = np.uint8(y * 255)
        return y
