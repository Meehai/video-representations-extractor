"""Depth dispresnet representation."""
import numpy as np
import torch as tr
from overrides import overrides
from matplotlib.cm import hot
import cv2

from .DispResNet import DispResNet
from ...representation import Representation, RepresentationOutput


def _preprocess(img: np.ndarray, size=None, trainSize=(256, 448), multiples=(64, 64)) -> tr.Tensor:
    if size is None:
        size = trainSize
    else:
        size = img.shape[:2]
    size = _closest_fit(size, multiples)
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = ((tr.from_numpy(img).unsqueeze(0) / 255 - 0.45) / 0.225)
    return img


def _postprocess(y: tr.Tensor, size=None, scale=(0, 2)) -> np.ndarray:
    if size is None:
        size = y.shape[2:4]
    y = y.cpu().numpy()[0, 0]
    dph = 1 / y

    dph = cv2.resize(dph, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    dph = (dph - scale[0]) / (scale[1] - scale[0])
    dph = np.clip(dph, 0, 1)

    return dph


def _closest_fit(size, multiples):
    return [round(size[i] / multiples[i]) * multiples[i] for i in range(len(multiples))]


class DepthDispResNet(Representation):
    def __init__(self, weightsFile: str, resNetLayers: int, trainHeight: int, trainWidth: int,
                 minDepth: int, maxDepth: int, **kwargs):
        super().__init__(**kwargs)
        self._setup()
        self.model = None
        self.weightsFile = weightsFile
        self.resNetLayers = resNetLayers
        self.multiples = (64, 64)  # is it 32 tho?
        self.trainSize = (trainHeight, trainWidth)
        self.scale = (minDepth, maxDepth)

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        raise NotImplementedError
        x = self.video[t]
        x_ = _preprocess(x, trainSize=self.trainSize, multiples=self.multiples)
        with tr.no_grad():
            y = self.model(x_)
        y = _postprocess(y, size=x.shape[:2], scale=self.scale)
        return y

    @overrides
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        y = x["data"] / x["data"].max()
        y = hot(y)[..., 0:3]
        y = np.uint8(y * 255)
        return y

    def _setup(self):
        if not self.model is None:
            return
        model = DispResNet(self.resNetLayers, False)
        weights = tr.load(self.weightsFile, map_location="cpu")
        model.load_state_dict(weights["state_dict"])
        model.eval()
        self.model = model
