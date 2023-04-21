import os
import numpy as np
import torch as tr
from typing import List
from overrides import overrides
from pathlib import Path
from media_processing_lib.image import image_resize
from torch import nn

from .Map2Map import EncoderMap2Map, DecoderMap2Map
from ...representation import Representation, RepresentationOutput

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")


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
    def __init__(self, numClasses: int, trainHeight: int, trainWidth: int, colorMap: List, weightsFile: str, **kwargs):
        self.model = None
        self.numClasses = numClasses
        assert len(colorMap) == numClasses, f"{colorMap} ({len(colorMap)}) vs {numClasses}"
        weightsFile = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/{weightsFile}").absolute()
        assert Path(weightsFile).exists(), f"Weights file '{weightsFile}' does not exist."
        self.weightsFile = weightsFile
        self.colorMap = colorMap
        self.trainHeight = trainHeight
        self.trainWidth = trainWidth
        super().__init__(**kwargs)

    @overrides
    def make(self, t: int) -> RepresentationOutput:
        frame = np.array(self.video[t])
        img = image_resize(frame, height=self.trainHeight, width=self.trainWidth, interpolation="bilinear")
        img = np.float32(img[None]) / 255
        tr_img = tr.from_numpy(img).to(device)
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

    @overrides
    def setup(self):

        model = MyModel(dIn=3, dOut=self.numClasses)
        data = tr.load(self.weightsFile, map_location="cpu")["state_dict"]
        params = model.state_dict()
        new_data = {}
        for k in data.keys():
            if k.startswith("model.0."):
                other = k.replace("model.0.", "encoder.")
            elif k.startswith("model.1."):
                other = k.replace("model.1.", "decoder.")
            else:
                assert False, (k, params.keys())
            new_data[other] = data[k]
        model.load_state_dict(new_data)
        self.model = model.eval().to(device)
