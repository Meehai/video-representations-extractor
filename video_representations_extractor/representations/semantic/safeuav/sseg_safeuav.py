import os
import numpy as np
import torch as tr
from typing import List
from overrides import overrides
from pathlib import Path
from media_processing_lib.image import image_resize
from ngclib.models.edges import SingleLink
from ngclib_cv.nodes import RGB, Semantic

from .Map2Map import EncoderMap2Map, DecoderMap2Map
from ...representation import Representation, RepresentationOutput

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

class _RGB(RGB):
    def get_encoder(self, outputNode):
        return EncoderMap2Map(dIn=self.num_dims)

    def get_decoder(self, inputNode):
        pass

class _Semantic(Semantic):
    def get_encoder(self, outputNode):
        pass

    def get_decoder(self, inputNode):
        return DecoderMap2Map(dOut=self.num_dims)


class SSegSafeUAV(Representation):
    def __init__(self, numClasses:int, trainHeight:int, trainWidth:int, colorMap:List, weightsFile:str, **kwargs):
        super().__init__(**kwargs)
        assert len(colorMap) == numClasses, f"{colorMap} ({len(colorMap)}) vs {numClasses}"
        weightsFile = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/{weightsFile}").absolute()
        assert Path(weightsFile).exists(), f"Weights file '{weightsFile}' does not exist."
        self.model = None
        self.numClasses = numClasses
        self.weightsFile = weightsFile
        self.colorMap = colorMap
        self.trainHeight = trainHeight
        self.trainWidth = trainWidth

    @overrides
    def make(self, t: int) -> RepresentationOutput:
        frame = np.array(self.video[t])
        img = image_resize(frame, height=self.trainHeight, width=self.trainWidth, interpolation="bilinear")
        img = np.float32(frame[None]) / 255
        tr_img = tr.from_numpy(img).to(device)
        with tr.no_grad():
            tr_res = self.model.forward(tr_img)[0]
        np_res = tr_res.to("cpu").numpy()
        res = np.argmax(np_res, axis=-1).astype(np.uint8)
        return res
    
    @overrides
    def makeImage(self, x: RepresentationOutput) -> np.ndarray:
        newImage = np.zeros((*x["data"].shape, 3), dtype=np.uint8)
        for i in range(self.numClasses):
            newImage[x["data"] == i] = self.colorMap[i]
        return newImage

    @overrides
    def setup(self):
        if not self.model is None:
            return

        rgbNode = _RGB(name="rgb")
        semanticNode = _Semantic(name="semantic", semantic_classes=self.numClasses, semantic_colors=self.colorMap)
        model = SingleLink(rgbNode, semanticNode)
        model.load_state_dict(tr.load(self.weightsFile))
        model.to(device)
        self.model = model
