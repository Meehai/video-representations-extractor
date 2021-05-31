import numpy as np
import cv2
import warnings
from typing import List, Tuple
from media_processing_lib.image import imgResize
from media_processing_lib.video import MPLVideo
from nwmodule.graph import Edge
from nwmodule.utilities import device
from cycleconcepts.nodes import RGB, Semantic
from cycleconcepts.models import SingleLinkGraph

from ..representation import Representation

class SSegSafeUAV(Representation):
	def __init__(self, baseDir:str, name:str, dependencies:List, video:MPLVideo, outShape:Tuple[int, int], \
		numClasses:int, trainHeight:int, trainWidth:int, colorMap:List, weightsFile:str):
		super().__init__(baseDir, name, dependencies, video, outShape)
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
		res = self.model.edges[0].model.npForward(img)[0]
		return res
	
	def makeImage(self, x):
		predicted_colormap = np.zeros((self.outShape[0], self.outShape[1], 3), dtype=np.uint8)
		label_indices = x["data"].argmax(axis=2)

		for current_prediction_idx in range(self.numClasses):
			predicted_colormap[np.nonzero(np.equal(label_indices,current_prediction_idx))] = \
				self.colorMap[current_prediction_idx]
		return predicted_colormap

	def setup(self):
		if not self.model is None:
			return

		rgbNode = RGB()
		semanticNode = Semantic(semanticClasses=list(range(self.numClasses)), semanticUseAllMetrics=False)
		model = SingleLinkGraph([
			Edge(rgbNode, semanticNode, blockGradients=False)
		])
		model.loadWeights(self.weightsFile, yolo=True)
		model.to(device)
		self.model = model