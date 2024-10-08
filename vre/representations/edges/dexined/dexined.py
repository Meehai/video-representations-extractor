"""Dexined representation."""
import numpy as np
import torch as tr
from overrides import overrides

from .model_dexined import DexiNed as Model
from ....representation import Representation, RepresentationOutput
from ....logger import vre_logger as logger
from ....utils import image_resize_batch, fetch_weights, load_weights

class DexiNed(Representation):
    """Dexined representation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = Model().eval()
        self.inference_height, self.inference_width = 512, 512 # fixed for this model

    # pylint: disable=arguments-differ
    @overrides(check_signature=False)
    def vre_setup(self):
        weights_data = load_weights(fetch_weights(__file__) / "dexined.pth")
        self.model.load_state_dict(weights_data)
        self.model = self.model.to(self.device)

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, RepresentationOutput] | None = None) -> RepresentationOutput:
        tr_frames = self._preprocess(frames, self.inference_height, self.inference_width)
        with tr.no_grad():
            y = self.model.forward(tr_frames)
        outs = self._postprocess(y)
        return RepresentationOutput(output=outs)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        x = np.repeat(np.expand_dims(repr_data.output, axis=-1), 3, axis=-1)
        return (x * 255).astype(np.uint8)

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return RepresentationOutput(output=image_resize_batch(repr_data.output, *new_size))

    @overrides
    def vre_free(self):
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()

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
        y_s_normed: tr.Tensor = 1 - (y_s - A) * (B - A)
        y_s_final = y_s_normed.mean(dim=1).cpu().numpy().astype(np.float32)
        return y_s_final
