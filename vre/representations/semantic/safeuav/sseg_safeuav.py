import os
import cv2
import numpy as np
import torch as tr
from overrides import overrides
from pathlib import Path
from torch import nn

from .Map2Map import EncoderMap2Map, DecoderMap2Map
from ....representation import Representation, RepresentationOutput


class MyModel(nn.Module):
    def __init__(self, dIn, dOut):
        super().__init__()
        self.encoder = EncoderMap2Map(dIn)
        self.decoder = DecoderMap2Map(dOut)

    def forward(self, x):
        y_encoder = self.encoder(x)
        y_decoder = self.decoder(y_encoder)
        return y_decoder


class SSegSafeUAV(Representation):
    def __init__(self, numClasses: int, trainHeight: int, trainWidth: int, colorMap: list[tuple[int, int, int]],
                 weights_file: str, device: str, **kwargs):
        self.model = None
        self.numClasses = numClasses
        assert len(colorMap) == numClasses, f"{colorMap} ({len(colorMap)}) vs {numClasses}"
        weights_file = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/{weights_file}").absolute()
        assert Path(weights_file).exists(), f"Weights file '{weights_file}' does not exist."
        self.weights_file = weights_file
        self.colorMap = colorMap
        self.trainHeight = trainHeight
        self.trainWidth = trainWidth
        self.device = device
        super().__init__(**kwargs)
        self._setup()

    @overrides
    def make(self, t: int) -> RepresentationOutput:
        frame = np.array(self.video[t])
        img = cv2.resize(frame, (self.trainWidth, self.trainHeight), interpolation=cv2.INTER_LINEAR)
        img = np.float32(img[None]) / 255
        tr_img = tr.from_numpy(img).to(self.device)
        with tr.no_grad():
            tr_res = self.model.forward(tr_img)[0]
        np_res = tr_res.to("cpu").numpy()
        res = np.argmax(np_res, axis=-1).astype(np.uint8)
        return res

    @overrides
    def make_image(self, x: RepresentationOutput) -> np.ndarray:
        newImage = np.zeros((*x["data"].shape, 3), dtype=np.uint8)
        for i in range(self.numClasses):
            newImage[x["data"] == i] = self.colorMap[i]
        return newImage

    def _setup(self):
        model = MyModel(dIn=3, dOut=self.numClasses)
        data = tr.load(self.weights_file, map_location="cpu")["state_dict"]
        params = model.state_dict()
        def convert(data: dict[str, tr.Tensor]) -> dict[str, tr.Tensor]:
            new_data = {}
            for k in data.keys():
                if k.startswith("model.0."):
                    other = k.replace("model.0.", "encoder.")
                elif k.startswith("model.1."):
                    other = k.replace("model.1.", "decoder.")
                else:
                    assert False, (k, params.keys())
                new_data[other] = data[k]
            return new_data

        model.load_state_dict(convert(data))
        self.model = model.eval().to(self.device)
