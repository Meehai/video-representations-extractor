#!/usr/bin/env python3
"""SafeUAV semanetic segmentation representation"""
import sys
from pathlib import Path
from overrides import overrides
import numpy as np
import torch as tr
from torch.nn import functional as F

from vre.vre_video import VREVideo
from vre.logger import vre_logger as logger
from vre.utils import MemoryData, image_read, image_write
from vre.representations import ReprOut, LearnedRepresentationMixin, ComputeRepresentationMixin
from vre_repository.weights_repository import fetch_weights
from vre_repository.semantic_segmentation import SemanticRepresentation

try:
    from .model import SafeUAV as Model
except ImportError:
    from model import SafeUAV as Model

class SafeUAV(SemanticRepresentation, LearnedRepresentationMixin, ComputeRepresentationMixin):
    """SafeUAV semantic segmentation representation"""
    def __init__(self, disk_data_argmax: bool, variant: str, **kwargs):
        LearnedRepresentationMixin.__init__(self)
        ComputeRepresentationMixin.__init__(self)
        self.variant = variant
        color_map = [[0, 255, 0], [0, 127, 0], [255, 255, 0], [255, 255, 255],
                     [255, 0, 0], [0, 0, 255], [0, 255, 255], [127, 127, 63]]
        classes = ["land", "forest", "residential", "road", "little-objects", "water", "sky", "hill"]
        SemanticRepresentation.__init__(self, classes=classes, color_map=color_map,
                                        disk_data_argmax=disk_data_argmax, **kwargs)
        self.model: Model | None = None
        self.cfg: dict | None = None
        self.statistics: dict[str, list[float]] | None = None
        self.output_dtype = "uint8" if disk_data_argmax else "float16"
        self._mean, self._std = None, None

    @property
    @overrides
    def n_channels(self) -> int:
        return len(self.classes)

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        h, w = self.cfg["model"]["hparams"]["data_shape"]["rgb"][1:3]
        cumsum = [0, *np.cumsum([x[0] for x in self.cfg["model"]["hparams"]["data_shape"].values()])]
        rgb_pos = self.cfg["data"]["parameters"]["task_names"].index("rgb")
        x = tr.zeros(len(ixs), self.model.encoder.d_in, h, w, device=self.device)
        tr_rgb = F.interpolate(tr.from_numpy(video[ixs]).permute(0, 3, 1, 2).to(self.device), size=(h, w))
        tr_rgb = (tr_rgb - self._mean) / self._std
        x[:, cumsum[rgb_pos]: cumsum[rgb_pos+1] ] = tr_rgb

        with tr.no_grad():
            y = self.model.forward(x)
        sema_pos = self.cfg["data"]["parameters"]["task_names"].index("semantic_output")
        y_rgb = y[:, cumsum[sema_pos]: cumsum[sema_pos+1]].permute(0, 2, 3, 1).cpu().numpy()
        self.data = ReprOut(frames=video[ixs], output=MemoryData(y_rgb), key=ixs)

    @staticmethod
    @overrides
    def weights_repository_links(**kwargs: dict) -> list[str]:
        assert (variant := kwargs["variant"]) != "testing", variant
        return [f"semantic_segmentation/safeuav/{variant}.ckpt"]

    @overrides
    def vre_setup(self, load_weights: bool = True):
        assert self.setup_called is False
        if self.variant == "testing":
            self.model = Model(in_channels=15, out_channels=15, num_filters=8)
            self.cfg = {
                "model": {
                    "hparams": {
                        "data_shape": {
                            "camera_normals_output": [3, 50, 50], "depth_output": [1, 50, 50],
                            "rgb": [3, 50, 50], "semantic_output": [8, 50, 50]
                        },
                    }
                },
                "data": {
                    "parameters": {
                        "task_names": ['depth_output', 'camera_normals_output', 'rgb', 'semantic_output']
                    }
                }
            }
            self.statistics = {
                "rgb": [[0.0, 0.0, 0.0],
                        [255.0, 255.0, 255.0],
                        [95.4980582891499, 103.0128696717979, 85.04794782640131],
                        [55.569107621001216, 58.133399225741776, 61.12564370234619]],
            }
        else:
            assert load_weights is True, load_weights
            if self.variant not in (variants := ("model_1M", "model_4M", "model_430k", "model_150k", "testing")):
                logger.warning(f"'{self.variant}' not in {variants}. Most likely a ckpt path.")
                assert Path(self.variant).exists(), self.variant
                ckpt = tr.load(self.variant, map_location="cpu")
            else:
                weights = fetch_weights(SafeUAV.weights_repository_links(variant=self.variant))[0]
                ckpt = tr.load(weights, map_location="cpu")
            self.cfg = ckpt["hyper_parameters"]["cfg"]
            self.statistics = ckpt["hyper_parameters"]["statistics"]
            self.model = Model(**self.cfg["model"]["parameters"])
            self.model.load_state_dict(ckpt["state_dict"])

        self._mean = tr.Tensor(self.statistics["rgb"][2]).reshape(1, 3, 1, 1).to(self.device)
        self._std = tr.Tensor(self.statistics["rgb"][3]).reshape(1, 3, 1, 1).to(self.device)
        self.model = self.model.eval().to(self.device)
        self.setup_called = True

    @overrides
    def vre_free(self):
        assert self.setup_called is True and self.model is not None, (self.setup_called, self.model is not None)
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
        self.model = None
        self.cfg = None
        self.statistics = None
        self.setup_called = False
        self._mean, self._std = None, None

if __name__ == "__main__":
    assert len(sys.argv) in (3, 4), ("Usage: python safeuav.py /path/to/img.png {model_430k/model_1M/model_4M/"
                                     "path_to_ckpt} [out_path]")
    img = image_read(sys.argv[1])
    model = SafeUAV(name="safeuav", disk_data_argmax=True, variant=sys.argv[2])
    model.vre_setup(load_weights=True)

    model.compute(img[None], [0])
    res_img = model.make_images(model.data)[0]
    out_path = Path(sys.argv[1]).parent / f"{Path(sys.argv[1]).stem}_res{Path(sys.argv[1]).suffix}"
    out_path = out_path if len(sys.argv) == 3 else sys.argv[3]
    image_write(res_img, out_path)
    logger.info(f"Stored prediction at '{out_path}'")
