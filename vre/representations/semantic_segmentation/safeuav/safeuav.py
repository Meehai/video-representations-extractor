"""SafeUAV semanetic segmentation representation"""
from pathlib import Path
import os
from overrides import overrides
import numpy as np
import torch as tr
from torch import nn
from torch.nn import functional as F

from .Map2Map import EncoderMap2Map, DecoderMap2Map
from ....representation import Representation, RepresentationOutput
from ....utils import image_resize_batch, VREVideo
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

class SafeUAV(Representation):
    """SafeUAV semanetic segmentation representation"""
    def __init__(self, num_classes: int, train_height: int, train_width: int,
                 color_map: list[tuple[int, int, int]], **kwargs):
        self.model: _SafeUavWrapper | None = None
        self.num_classes = num_classes
        assert len(color_map) == num_classes, f"{color_map} ({len(color_map)}) vs {num_classes}"
        self.color_map = color_map
        self.train_height = train_height
        self.train_width = train_width
        self.device = "cpu"
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

        weights_dir = Path(f"{os.environ['VRE_WEIGHTS_DIR']}").absolute()
        weights_file_abs = weights_dir / weights_file
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
        # we need to argmax here because VRE expects [0:1] values for float or a uint8 for semantic (for now?)
        np_pred_argmax = np.argmax(np_pred, axis=-1).astype(np.uint8)
        return np_pred_argmax

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        new_images = np.zeros((*repr_data.shape, 3), dtype=np.uint8)
        # TODO: use video[t] and return in make() the logits/softmax, not argmax for better upscale
        for i in range(self.num_classes):
            new_images[repr_data == i] = self.color_map[i]
        # order=0 is nearest neighbor
        new_images_rsz = image_resize_batch(new_images, height=frames.shape[1], width=frames.shape[2], order=0)
        return new_images_rsz
