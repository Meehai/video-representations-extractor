import os
import gdown
import numpy as np
import torch as tr
import cv2
from pathlib import Path
from typing import List
from overrides import overrides

from .model_dexined import DexiNed as Model
from ....representation import Representation, RepresentationOutput
from ....logger import logger

def _preprocess(images: np.ndarray) -> np.ndarray:
    assert len(images.shape) == 4, images.shape
    logger.debug2(f"Original shape: {images.shape}")
    images_resize = np.array([cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR) for image in images])
    mean_pixel_values = [103.939, 116.779, 123.68]
    images_norm = images_resize.astype(np.float32) - mean_pixel_values
    # N, H, W, C -> N, C, H, W
    images_norm = images_norm.transpose((0, 3, 1, 2))
    return images_norm


def _postprocess(y: list[tr.Tensor]) -> np.ndarray:
    # y :: (B, T, H, W)
    y = tr.cat(y, dim=1)
    assert len(y.shape) == 4, y.shape
    y_s = y.sigmoid()
    A = y_s.min(dim=-1, keepdims=True)[0].min(dim=-2, keepdims=True)[0]
    B = y_s.max(dim=-1, keepdims=True)[0].max(dim=-2, keepdims=True)[0]
    y_s_normed = 1 - (y_s - A) * (B - A)
    y_s_final = y_s_normed.mean(dim=1).cpu().numpy()
    return y_s_final


class DexiNed(Representation):
    def __init__(self, device: str, **kwargs):
        self.model = None
        self.device = device
        assert tr.cuda.is_available() or self.device == "cpu", "CUDA not available"
        super().__init__(**kwargs)
        self._setup()

    def _setup(self):
        # our backup weights
        weights_file = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/dexined.pth").absolute()
        url_weights = "https://drive.google.com/u/0/uc?id=1oT1iKdRRKJpQO-DTYWUnZSK51QnJ-mnP"

        if not weights_file.exists():
            logger.debug(f"Downloading weights for dexined from {url_weights}")
            gdown.download(url_weights, str(weights_file))

        model = Model()
        model.load_state_dict(tr.load(weights_file, map_location="cpu"))
        model.eval()
        logger.debug2(f"Loaded weights from '{weights_file}'")
        self.model = model.to(self.device)

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        frames = np.array(self.video[t])
        A = _preprocess(frames)
        trA = tr.from_numpy(A).float().to(self.device)
        with tr.no_grad():
            trB = self.model.forward(trA)
        C = _postprocess(trB)
        return C

    @overrides
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        x = np.repeat(np.expand_dims(x, axis=-1), 3, axis=-1)
        return np.uint8(x * 255)
