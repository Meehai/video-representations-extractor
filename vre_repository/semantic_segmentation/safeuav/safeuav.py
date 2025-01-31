"""SafeUAV semanetic segmentation representation"""
import sys
from overrides import overrides
import numpy as np
import torch as tr
from torch import nn
from torch.nn import functional as F

from vre.utils import VREVideo, MemoryData, image_read
from vre.logger import vre_logger as logger
from vre.representations import ReprOut, LearnedRepresentationMixin, ComputeRepresentationMixin
from vre_repository.weights_repository import fetch_weights
from vre_repository.semantic_segmentation import SemanticRepresentation

try:
    from .model import SafeUAV as Model
except ImportError:
    from model import SafeUAV as Model

class SafeUAV(SemanticRepresentation, LearnedRepresentationMixin, ComputeRepresentationMixin):
    """SafeUAV semantic segmentation representation"""
    def __init__(self, num_classes: int, color_map: list[tuple[int, int, int]],
                 disk_data_argmax: bool, variant: str, **kwargs):
        LearnedRepresentationMixin.__init__(self)
        ComputeRepresentationMixin.__init__(self)
        SemanticRepresentation.__init__(self, classes=list(range(num_classes)), color_map=color_map,
                                        disk_data_argmax=disk_data_argmax, **kwargs)
        self.variant = variant
        self.model: Model | None = None
        self.output_dtype = "uint8" if disk_data_argmax else "float16"

    @property
    @overrides
    def n_channels(self) -> int:
        return len(self.classes)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        tr_frames = tr.from_numpy(video[ixs]).to(self.device)
        frames_norm = tr_frames.permute(0, 3, 1, 2) / 255
        frames_resized = F.interpolate(frames_norm, (self.train_height, self.train_width), mode="bilinear")
        with tr.no_grad():
            prediction = self.model.forward(frames_resized)
        np_pred = prediction.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
        self.data = ReprOut(frames=video[ixs], output=MemoryData(np_pred), key=ixs)

    @staticmethod
    @overrides
    def weights_repository_links(**kwargs: dict) -> list[str]:
        assert (variant := kwargs["variant"]) != "testing", variant
        return [f"semantic_segmentation/safeuav/{variant}.ckpt"]

    @overrides
    def vre_setup(self, load_weights: bool = True):
        assert self.setup_called is False
        assert self.variant in ("model_4M", "testing"), self.variant

        if self.variant == "testing":
            num_filters = 8
        else:
            num_filters = 32
        self.model = Model(in_channels=15, out_channels=self.n_channels, num_filters=num_filters)

        if load_weights:
            ckpt = tr.load(fetch_weights(SafeUAV.weights_repository_links(variant=self.variant))[0])
            self.model.load_state_dict(ckpt["state_dict"])

        self.model = self.model.eval().to(self.device)
        self.setup_called = True

    @overrides
    def vre_free(self):
        assert self.setup_called is True and self.model is not None, (self.setup_called, self.model is not None)
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
        self.model = None
        self.setup_called = False

if __name__ == "__main__":
    img = image_read(sys.argv[1])
    color_map = [[0, 255, 0], [0, 127, 0], [255, 255, 0], [255, 255, 255],
                    [255, 0, 0], [0, 0, 255], [0, 255, 255], [127, 127, 63]]
    model = SafeUAV(name="safeuav", num_classes=8, color_map=color_map, disk_data_argmax=True, variant="model_4M")
    model.vre_setup(load_weights=True)

    breakpoint()
