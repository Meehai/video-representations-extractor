import numpy as np
import cv2
import warnings
from typing import List, Tuple, Union
from media_processing_lib.image import imgResize
from media_processing_lib.video import MPLVideo
from nwmodule.graph import Edge
from nwmodule.utilities import device
# from ngclib.nodes import RGB, Semantic
from ngclib.models import SingleLink

from .rgb import RGB
from .semantic import Semantic
from ..representation import Representation

class SSegSafeUAV(Representation):
	def __init__(self, name:str, dependencies:List[Union[str, Representation]], saveResults:str, \
		dependencyAliases:List[str], numClasses:int, trainHeight:int, trainWidth:int, colorMap:List, weightsFile:str):
		super().__init__(name, dependencies, saveResults, dependencyAliases)
		assert len(colorMap) == numClasses, "%s vs %d" % (colorMap, numClasses)
		self.model = None
		self.numClasses = numClasses
		self.weightsFile = weightsFile
		self.colorMap = colorMap
		self.trainHeight = trainHeight
		self.trainWidth = trainWidth

	def make(self, t):
		frame = np.array(self.video[t])
		img = imgResize(frame, height=self.trainHeight, width=self.trainWidth, interpolation="bilinear")
		img = np.float32(frame[None]) / 255
		res = self.model.npForward(img)[0]
		res = np.argmax(res, axis=-1).astype(np.uint8)
		return res
	
	def makeImage(self, x:np.ndarray) -> np.ndarray:
		newImage = np.zeros((*x["data"].shape, 3), dtype=np.uint8)
		for i in range(self.numClasses):
			newImage[x["data"] == i] = self.colorMap[i]
		return newImage

	def setup(self):
		if not self.model is None:
			return

		rgbNode = RGB()
		semanticNode = Semantic(semanticClasses=list(range(self.numClasses)), semanticColors=self.colorMap, \
			name="semantic")
		# model = SingleLinkGraph([
		# 	Edge(rgbNode, semanticNode, blockGradients=False)
		# ])
		model = SingleLink(rgbNode, semanticNode)
		model.loadWeights(self.weightsFile, yolo=True)
		model.to(device)
		self.model = model
