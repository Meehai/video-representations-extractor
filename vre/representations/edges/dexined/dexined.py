"""Dexined representation."""
import numpy as np
import torch as tr
from overrides import overrides

from .model_dexined import DexiNed as Model
from ....representation import Representation, RepresentationOutput
from ....logger import logger
from ....utils import image_resize_batch, gdown_mkdir, VREVideo, get_weights_dir

class DexiNed(Representation):
    """Dexined representation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: Model = None
        self._setup()
        self.device = "cpu"
        self.inference_height, self.inference_width = 512, 512 # fixed for this model

    def _setup(self):
        tr.manual_seed(42)
        self.model = Model().eval().to("cpu")

    # pylint: disable=arguments-differ
    @overrides(check_signature=False)
    def vre_setup(self, video: VREVideo, device: str):
        assert tr.cuda.is_available() or device == "cpu", "CUDA not available"
        self.device = device
        weights_file = get_weights_dir() / "dexined.pth"
        url_weights = "https://drive.google.com/u/0/uc?id=1oT1iKdRRKJpQO-DTYWUnZSK51QnJ-mnP" # our backup weights

        if not weights_file.exists():
            gdown_mkdir(url_weights, weights_file)

        logger.info(f"Loading weights from '{weights_file}'")
        weights_data = tr.load(weights_file, map_location="cpu")
        self.model.load_state_dict(weights_data)
        self.model = self.model.to(self.device)
        logger.debug2(f"Loaded weights from '{weights_file}'")

    @overrides
    def make(self, frames: np.ndarray) -> RepresentationOutput:
        tr_frames = self._preprocess(frames, self.inference_height, self.inference_width)
        with tr.no_grad():
            y = self.model.forward(tr_frames)
        outs = self._postprocess(y)
        return outs

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        x = np.repeat(np.expand_dims(repr_data, axis=-1), 3, axis=-1)
        return (x * 255).astype(np.uint8)

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return image_resize_batch(repr_data, *new_size)

    def _preprocess(self, images: np.ndarray, height: int, width: int) -> tr.Tensor:
        assert len(images.shape) == 4, images.shape
        logger.debug2(f"Original shape: {images.shape}")
        images_resize = image_resize_batch(images, height, width)
        mean_pixel_values = [103.939, 116.779, 123.68]
        images_norm = images_resize.astype(np.float32) - mean_pixel_values
        # N, H, W, C -> N, C, H, W
        images_norm = images_norm.transpose((0, 3, 1, 2))
        tr_images_norm = tr.from_numpy(images_norm).to(self.device).float()
        return tr_images_norm

    def _postprocess(self, y: list[tr.Tensor]) -> np.ndarray:
        # y :: (B, T, H, W)
        y_cat = tr.cat(y, dim=1)
        assert len(y_cat.shape) == 4, y_cat.shape
        y_s = y_cat.sigmoid()
        A = y_s.min(dim=-1, keepdims=True)[0].min(dim=-2, keepdims=True)[0]
        B = y_s.max(dim=-1, keepdims=True)[0].max(dim=-2, keepdims=True)[0]
        y_s_normed = 1 - (y_s - A) * (B - A)
        y_s_final = y_s_normed.mean(dim=1).cpu().numpy().astype(np.float16)
        return y_s_final
