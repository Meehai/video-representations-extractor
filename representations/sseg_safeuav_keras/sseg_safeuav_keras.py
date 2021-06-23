import numpy as np
import cv2
import warnings
from typing import List, Tuple
from media_processing_lib.video import MPLVideo
from .safeuav import get_unet_MDCB_with_deconv_layers
from ..representation import Representation

warnings.filterwarnings("ignore")

def get_disjoint_prediction_fast(prediction_map):
	height, width, nChs = prediction_map.shape
	position = np.argmax(prediction_map, axis=2)
	values = np.max(prediction_map, axis=2)
	disjoint_map = np.zeros_like(prediction_map)
	xx, yy = np.meshgrid(np.arange(height), np.arange(width))
	disjoint_map[xx, yy, position.transpose()] =  values.transpose()
	return disjoint_map

class SSegSafeUAVKeras(Representation):
	def __init__(self, name:str, dependencies:List[Representation], dependencyAliases:List[str], \
		numClasses:int, colorMap:List, trainHeight:int, trainWidth:int, init_nb:int, weightsFile:str):
		super().__init__(name, dependencies, dependencyAliases)
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
		pred = self.model.predict(img)
		result = np.array(pred[0], dtype=np.float32)
		return result

	def makeImage(self, x:np.ndarray) -> np.ndarray:
		predicted_label = get_disjoint_prediction_fast(x["data"])
		predicted_colormap = np.zeros((self.outShape[0], self.outShape[1], 3), dtype=np.uint8)
		label_indices = predicted_label.argmax(axis=2)

		for current_prediction_idx in range(self.numClasses):
			predicted_colormap[np.nonzero(np.equal(label_indices,current_prediction_idx))] = \
				self.colorMap[current_prediction_idx]
		return predicted_colormap

	def setup(self):
		if not self.model is None:
			return
		model = get_unet_MDCB_with_deconv_layers(input_shape=(self.trainHeight, self.trainWidth, 3), \
			init_nb=self.init_nb, num_classes=self.numClasses)
		model.load_weights(filepath=self.weightsFile)
		self.model = model
