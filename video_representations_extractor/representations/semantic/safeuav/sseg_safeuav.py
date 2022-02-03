import numpy as np
import torch as tr
from typing import List
from overrides import overrides
from pathlib import Path
from media_processing_lib.image import image_resize
from ngclib.models.edges import SingleLink

from .rgb import RGB
from .semantic import Semantic
from ...representation import Representation, RepresentationOutput

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

class SSegSafeUAV(Representation):
	def __init__(self, numClasses:int, trainHeight:int, trainWidth:int, colorMap:List, weightsFile:str, **kwargs):
		super().__init__(**kwargs)
		assert len(colorMap) == numClasses, f"{colorMap} ({len(colorMap)}) vs {numClasses}"
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
		res = self.model.npForward(img)[0]
		res = np.argmax(res, axis=-1).astype(np.uint8)
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

		rgbNode = RGB()
		semanticNode = Semantic(semanticClasses=list(range(self.numClasses)), semanticColors=self.colorMap, \
			name="semantic")
		model = SingleLink(rgbNode, semanticNode)
		model.loadWeights(self.weightsFile, yolo=True)
		model.to(device)
		self.model = model
