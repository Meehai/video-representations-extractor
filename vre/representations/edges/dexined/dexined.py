import os
import gdown
import numpy as np
import torch as tr
from pathlib import Path
from overrides import overrides

from .model_dexined import DexiNed as Model
from ....representation import Representation, RepresentationOutput
from ....logger import logger
from ....utils import image_resize

def _preprocess(images: np.ndarray, height: int, width: int) -> np.ndarray:
    assert len(images.shape) == 4, images.shape
    logger.debug2(f"Original shape: {images.shape}")
    images_resize = np.array([image_resize(image, height, width) for image in images])
    mean_pixel_values = [103.939, 116.779, 123.68]
    images_norm = images_resize.astype(np.float32) - mean_pixel_values
    # N, H, W, C -> N, C, H, W
    images_norm = images_norm.transpose((0, 3, 1, 2))
    return images_norm


def _postprocess(y: list[tr.Tensor]) -> np.ndarray:
    # y :: (B, T, H, W)
    y_cat = tr.cat(y, dim=1)
    assert len(y_cat.shape) == 4, y_cat.shape
    y_s = y_cat.sigmoid()
    A = y_s.min(dim=-1, keepdims=True)[0].min(dim=-2, keepdims=True)[0]
    B = y_s.max(dim=-1, keepdims=True)[0].max(dim=-2, keepdims=True)[0]
    y_s_normed = 1 - (y_s - A) * (B - A)
    y_s_final = y_s_normed.mean(dim=1).cpu().numpy()
    return y_s_final


class DexiNed(Representation):
    def __init__(self, inference_height: int, inference_width: int, **kwargs):
        self.model: Model = None
        super().__init__(**kwargs)
        self._setup()
        self.device = "cpu"
        self.inference_height = inference_height
        self.inference_width = inference_width

    def _setup(self):
        self.model = Model().eval()

    @overrides(check_signature=False)
    def vre_setup(self, device: str):
        assert tr.cuda.is_available() or device == "cpu", "CUDA not available"
        self.device = device
        # our backup weights
        weights_file = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/dexined.pth").absolute()
        url_weights = "https://drive.google.com/u/0/uc?id=1oT1iKdRRKJpQO-DTYWUnZSK51QnJ-mnP"

        if not weights_file.exists():
            logger.debug(f"Downloading weights for dexined from {url_weights}")
            gdown.download(url_weights, str(weights_file))

        self.model.load_state_dict(tr.load(weights_file, map_location="cpu"))
        self.model = self.model.to(self.device)
        logger.debug2(f"Loaded weights from '{weights_file}'")

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        frames = np.array(self.video[t])
        A = _preprocess(frames, self.inference_height, self.inference_width)
        trA = tr.from_numpy(A).float().to(self.device)
        with tr.no_grad():
            trB = self.model.forward(trA)
        C = _postprocess(trB)
        return C

    @overrides
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        x = np.repeat(np.expand_dims(x, axis=-1), 3, axis=-1)
        return (x * 255).astype(np.uint8)
