import numpy as np
import cv2
import warnings
from typing import List, Tuple
from media_processing_lib.video import MPLVideo
from .safeuav import get_unet_MDCB_with_deconv_layers
from ...representation import Representation

warnings.filterwarnings("ignore")

def get_disjoint_prediction_fast(prediction_map):
	height, width, nChs = prediction_map.shape
	position = np.argmax(prediction_map, axis=2)
	values = np.max(prediction_map, axis=2)
	disjoint_map = np.zeros_like(prediction_map)
	xx, yy = np.meshgrid(np.arange(height), np.arange(width))
	disjoint_map[xx, yy, position.transpose()] = values.transpose()
	return disjoint_map

class SSegSafeUAVKeras(Representation):
	def __init__(self, name:str, dependencies:List[Representation], saveResults:str, dependencyAliases:List[str], \
		numClasses:int, colorMap:List, trainHeight:int, trainWidth:int, init_nb:int, weightsFile:str):
		super().__init__(name, dependencies, saveResults, dependencyAliases)
		assert len(colorMap) == numClasses, "%s vs %d" % (colorMap, numClasses)
		self.model = None
		self.numClasses = numClasses
		self.trainHeight = trainHeight
		self.trainWidth = trainWidth
		self.init_nb = init_nb
		self.weightsFile = weightsFile
		self.colorMap = colorMap

	def make(self, t:int) -> np.ndarray:
		orig_img = self.video[t]
		input_img = cv2.resize(orig_img, (self.trainWidth, self.trainHeight))
		img = (np.float32(input_img) / 255)[None]
		pred = self.model.predict(img)[0]
		result = pred.argmax(axis=-1).astype(np.uint8)
		return result

	def makeImage(self, x:np.ndarray) -> np.ndarray:
		newImage = np.zeros((*x["data"].shape, 3), dtype=np.uint8)
		for i in range(self.numClasses):
			newImage[x["data"] == i] = self.colorMap[i]
		return newImage

	def setup(self):
		if not self.model is None:
			return
		model = get_unet_MDCB_with_deconv_layers(input_shape=(self.trainHeight, self.trainWidth, 3), \
			init_nb=self.init_nb, num_classes=self.numClasses)
		model.load_weights(filepath=self.weightsFile)
		self.model = model
