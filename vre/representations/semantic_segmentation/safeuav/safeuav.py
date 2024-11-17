"""SafeUAV semanetic segmentation representation"""
from overrides import overrides
import numpy as np
import torch as tr
from torch import nn
from torch.nn import functional as F

from vre.utils import (
    image_resize_batch, fetch_weights, vre_load_weights, colorize_semantic_segmentation, VREVideo, MemoryData)
from vre.logger import vre_logger as logger
from vre.representations import (
    Representation, ReprOut, LearnedRepresentationMixin, ComputeRepresentationMixin, NpIORepresentation)
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

class SafeUAV(Representation, LearnedRepresentationMixin, ComputeRepresentationMixin, NpIORepresentation):
    """SafeUAV semantic segmentation representation"""
    def __init__(self, num_classes: int, train_height: int, train_width: int, color_map: list[tuple[int, int, int]],
                 semantic_argmax_only: bool, weights_file: str | None = None, **kwargs):
        Representation.__init__(self, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        self.num_classes = num_classes
        assert len(color_map) == num_classes, f"{color_map} ({len(color_map)}) vs {num_classes}"
        self.color_map = color_map
        self.train_height = train_height
        self.train_width = train_width
        self.semantic_argmax_only = semantic_argmax_only
        self.weights_file = weights_file
        self.classes = list(range(num_classes))
        self.model: _SafeUavWrapper | None = None
        self.output_dtype = "uint8" if semantic_argmax_only else "float16"

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
    def make_images(self) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        res = []
        frames_rsz = image_resize_batch(self.data.frames, *self.data.output.shape[1:3])
        for img, pred in zip(frames_rsz, self.data.output):
            _pred: np.ndarray = pred if self.semantic_argmax_only else pred.argmax(-1)
            res.append(colorize_semantic_segmentation(_pred, self.classes, self.color_map, img))
        res = np.stack(res)
        return res

    @overrides
    def vre_setup(self, load_weights: bool = True):
        assert self.setup_called is False
        self.model = _SafeUavWrapper(ch_in=3, ch_out=self.num_classes)

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
