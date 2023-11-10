import os
import numpy as np
import torch as tr
from pathlib import Path
from overrides import overrides

from .model_dexined import DexiNed as Model
from ....representation import Representation, RepresentationOutput
from ....logger import logger
from ....utils import image_resize_batch, gdown_mkdir

def _preprocess(images: np.ndarray, height: int, width: int) -> np.ndarray:
    assert len(images.shape) == 4, images.shape
    logger.debug2(f"Original shape: {images.shape}")
    images_resize = image_resize_batch(images, height, width)
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
        tr.manual_seed(42)
        self.model = Model().eval().to("cpu")

    @overrides(check_signature=False)
    def vre_setup(self, device: str):
        assert tr.cuda.is_available() or device == "cpu", "CUDA not available"
        self.device = device
        # our backup weights
        weights_file = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/dexined.pth").absolute()
        url_weights = "https://drive.google.com/u/0/uc?id=1oT1iKdRRKJpQO-DTYWUnZSK51QnJ-mnP"

        if not weights_file.exists():
            gdown_mkdir(url_weights, weights_file)

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
    def make_images(self, t: slice, x: np.ndarray, extra: dict | None) -> np.ndarray:
        x = np.repeat(np.expand_dims(x, axis=-1), 3, axis=-1)
        x_rsz = image_resize_batch(x, height=self.video.frame_shape[0], width=self.video.frame_shape[1])
        return (x_rsz * 255).astype(np.uint8)
