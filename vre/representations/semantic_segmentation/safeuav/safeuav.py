"""SafeUAV semanetic segmentation representation"""
from overrides import overrides
import numpy as np
import torch as tr
from torch import nn
from torch.nn import functional as F

from .Map2Map import EncoderMap2Map, DecoderMap2Map
from ....representation import Representation, RepresentationOutput
from ....utils import image_resize_batch, VREVideo, get_weights_dir
from ....logger import logger

class _SafeUavWrapper(nn.Module):
    """Wrapper. TODO: Replace with nn.Sequential"""
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        tr.manual_seed(42)
        self.encoder = EncoderMap2Map(ch_in)
        self.decoder = DecoderMap2Map(ch_out)

    def forward(self, x):
        """forward function"""
        y_encoder = self.encoder(x)
        y_decoder = self.decoder(y_encoder)
        return y_decoder

def _convert(data: dict[str, tr.Tensor]) -> dict[str, tr.Tensor]:
    new_data = {}
    for k in data.keys():
        if k.startswith("model.0."):
            other = k.replace("model.0.", "encoder.")
        elif k.startswith("model.1."):
            other = k.replace("model.1.", "decoder.")
        else:
            assert False, k
        new_data[other] = data[k]
    return new_data

# TODO: make semantic_argmax_only not optional
class SafeUAV(Representation):
    """SafeUAV semantic segmentation representation"""
    def __init__(self, num_classes: int, train_height: int, train_width: int,
                 color_map: list[tuple[int, int, int]], semantic_argmax_only: bool = True, **kwargs):
        self.num_classes = num_classes
        assert len(color_map) == num_classes, f"{color_map} ({len(color_map)}) vs {num_classes}"
        self.color_map = color_map
        self.train_height = train_height
        self.train_width = train_width
        self.device = "cpu"
        self.semantic_argmax_only = semantic_argmax_only
        super().__init__(**kwargs)
        self.model = _SafeUavWrapper(ch_in=3, ch_out=self.num_classes).eval().to("cpu")

    @overrides(check_signature=False)
    def vre_setup(self, video: VREVideo, weights_file: str | None, device: str, **kwargs):
        # pylint: disable=arguments-differ
        self.device = device
        if weights_file is None:
            self.model = self.model.eval().to(self.device)
            logger.warning("No weights file provided, using random weights.")
            return

        weights_file_abs = get_weights_dir() / weights_file
        assert weights_file_abs.exists(), f"Weights file '{weights_file_abs}' does not exist."
        data = tr.load(weights_file_abs, map_location="cpu")["state_dict"]
        self.model.load_state_dict(_convert(data))
        self.model = self.model.eval().to(self.device)

    @overrides
    def make(self, frames: np.ndarray) -> RepresentationOutput:
        tr_frames = tr.from_numpy(frames).to(self.device)
        frames_norm = tr_frames.permute(0, 3, 1, 2) / 255
        frames_resized = F.interpolate(frames_norm, (self.train_height, self.train_width), mode="bilinear")
        with tr.no_grad():
            prediction = self.model.forward(frames_resized)
        np_pred = prediction.permute(0, 2, 3, 1).cpu().numpy()
        y_out = np.argmax(np_pred, axis=-1).astype(np.uint8) if self.semantic_argmax_only else np_pred
        return y_out

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        # TODO: use visualizer from M2F.
        repr_data = repr_data if self.semantic_argmax_only else repr_data.argmax(-1)
        new_images = np.zeros((*repr_data.shape, 3), dtype=np.uint8)
        for i in range(self.num_classes):
            new_images[repr_data == i] = self.color_map[i]
        return new_images

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        interpolation = "nearest" if self.semantic_argmax_only else "bilinear"
        return image_resize_batch(repr_data, *new_size, interpolation=interpolation)
