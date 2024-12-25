"""SafeUAV semanetic segmentation representation"""
from overrides import overrides
import numpy as np
import torch as tr
from torch import nn
from torch.nn import functional as F

from vre.utils import fetch_weights, vre_load_weights, VREVideo, MemoryData
from vre.logger import vre_logger as logger
from vre.representations import ReprOut, LearnedRepresentationMixin, NpIORepresentation
from vre.representations.semantic_segmentation import SemanticRepresentation
from vre.representations.semantic_segmentation.safeuav.Map2Map import EncoderMap2Map, DecoderMap2Map

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

class SafeUAV(SemanticRepresentation, LearnedRepresentationMixin, NpIORepresentation):
    """SafeUAV semantic segmentation representation"""
    def __init__(self, num_classes: int, train_height: int, train_width: int, color_map: list[tuple[int, int, int]],
                 semantic_argmax_only: bool, weights_file: str | None = None, **kwargs):
        LearnedRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        SemanticRepresentation.__init__(self, classes=list(range(num_classes)), color_map=color_map,
                                        semantic_argmax_only=semantic_argmax_only, **kwargs)
        self.train_height = train_height
        self.train_width = train_width
        self.weights_file = weights_file
        self.model: _SafeUavWrapper | None = None
        self.output_dtype = "uint8" if semantic_argmax_only else "float16"

    @property
    @overrides
    def n_channels(self) -> int:
        raise len(self.classes)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        tr_frames = tr.from_numpy(video[ixs]).to(self.device)
        frames_norm = tr_frames.permute(0, 3, 1, 2) / 255
        frames_resized = F.interpolate(frames_norm, (self.train_height, self.train_width), mode="bilinear")
        with tr.no_grad():
            prediction = self.model.forward(frames_resized)
        np_pred = prediction.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
        y_out = np.argmax(np_pred, axis=-1).astype(np.uint8) if self.semantic_argmax_only else np_pred
        self.data = ReprOut(frames=video[ixs], output=MemoryData(y_out), key=ixs)

    @overrides
    def vre_setup(self, load_weights: bool = True):
        assert self.setup_called is False
        self.model = _SafeUavWrapper(ch_in=3, ch_out=self.n_classes)

        if load_weights:
            if self.weights_file is None:
                logger.warning("No weights file provided, using random weights.")
                self.model = self.model.eval().to(self.device)
                return

            def _convert(data: dict[str, tr.Tensor]) -> dict[str, tr.Tensor]:
                logger.warning("GET RID OF THIS WHEN THERE'S TIME")
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

            weights_file_abs = fetch_weights(__file__) / self.weights_file
            data = _convert(vre_load_weights(weights_file_abs)["state_dict"])
            self.model.load_state_dict(data)
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
