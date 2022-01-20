import numpy as np
import torch as tr
from overrides import overrides
from media_processing_lib.image import image_resize
from matplotlib.cm import hot

from .DispResNet import DispResNet
from ..representation import Representation

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

def preprocessImage(img, size=None, trainSize=(256, 448), multiples=(64, 64)):
	if size is None:
		size = trainSize
	else:
		size = img.shape[:2]
	size = closest_fit(size, multiples)
	img = image_resize(img, height=size[0], width=size[1])
	# img = imresize(img, closest_fit(size, multiples))
	img = img.astype(np.float32)
	img = np.transpose(img, (2, 0, 1))
	img = ((tr.from_numpy(img).unsqueeze(0) / 255 - 0.45) / 0.225).to(device)
	return img

def postprocessImage(y, size=None, scale=(0, 2)):
	if size is None:
		size = y.shape[2:4]
	y = y.cpu().numpy()[0, 0]
	dph = 1 / y

	dph = image_resize(dph, height=size[0], width=size[1], onlyUint8=False)
	dph = (dph - scale[0]) / (scale[1] - scale[0])
	dph = np.clip(dph, 0, 1)

	return dph

def closest_fit(size, multiples):
	return [round(size[i] / multiples[i]) * multiples[i] for i in range(len(multiples))]


class DepthDispResNet(Representation):
	def __init__(self, weightsFile:str, resNetLayers:int, trainHeight:int, \
			trainWidth:int, minDepth:int, maxDepth:int, **kwargs):
		super().__init__(**kwargs)
		self.model = None
		self.weightsFile = weightsFile
		self.resNetLayers = resNetLayers
		self.multiples = (64, 64)   # is it 32 tho?
		self.trainSize = (trainHeight, trainWidth)
		self.scale = (minDepth, maxDepth)

	@overrides
	def make(self, t: int) -> np.ndarray:
		x = self.video[t]
		x_ = preprocessImage(x, trainSize=self.trainSize, multiples=self.multiples)
		with tr.no_grad():
			y = self.model(x_)
		y = postprocessImage(y, size=x.shape[:2], scale=self.scale)
		return y

	@overrides
	def makeImage(self, x: np.ndarray) -> np.ndarray:
		y = x["data"] / x["data"].max()
		y = hot(y)[..., 0:3]
		y = np.uint8(y * 255)
		return y

	@overrides
	def setup(self):
		if not self.model is None:
			return
		model = DispResNet(self.resNetLayers, False).to(device)
		weights = tr.load(self.weightsFile, map_location=device)
		model.load_state_dict(weights["state_dict"])
		model.eval()
		self.model = model
