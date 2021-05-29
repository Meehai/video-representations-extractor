import numpy as np
import torch
from media_processing_lib.image import imgResize
from media_processing_lib.video import MPLVideo
from nwmodule.utilities import device
from matplotlib.cm import hot
from typing import Dict

from .DispResNet import DispResNet
from ..representation import Representation

def preprocessImage(img, size=None, trainSize=(256, 448), multiples=(64, 64), device=torch.device('cuda')):
	if size is None:
		size = trainSize
	else:
		size = img.shape[:2]
	size = closest_fit(size, multiples)
	img = imgResize(img, height=size[0], width=size[1])
	# img = imresize(img, closest_fit(size, multiples))
	img = img.astype(np.float32)
	img = np.transpose(img, (2, 0, 1))
	img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.45) / 0.225).to(device)
	return img

def postprocessImage(y, size=None, scale=(0, 2)):
	if size is None:
		size = y.shape[2:4]
	y = y.cpu().numpy()[0, 0]
	dph = 1 / y

	dph = imgResize(dph, height=size[0], width=size[1], onlyUint8=False)
	dph = (dph - scale[0]) / (scale[1] - scale[0])
	dph = np.clip(dph, 0, 1)

	return dph

def closest_fit(size, multiples):
	return [round(size[i] / multiples[i]) * multiples[i] for i in range(len(multiples))]


class DepthJiaw(Representation):
	def __init__(self, baseDir, name, dependencies, video, outShape, weightsFile:str, resNetLayers:int, \
		trainHeight:int, trainWidth:int, minDepth:int, maxDepth:int):
		super().__init__(baseDir, name, dependencies, video, outShape)
		self.model = None
		self.weightsFile = weightsFile
		self.resNetLayers = resNetLayers
		self.multiples = (64, 64)   # is it 32 tho?
		self.trainSize = (trainHeight, trainWidth)
		self.scale = (minDepth, maxDepth)

	def make(self, t:int) -> np.ndarray:
		x = self.video[t]
		x_ = preprocessImage(x, trainSize=self.trainSize, multiples=self.multiples, device=device)
		with torch.no_grad():
			y = self.model(x_)
		y = postprocessImage(y, size=x.shape[:2], scale=self.scale)
		return y

	def makeImage(self, x):
		y = hot(x)[..., 0:3]
		y = np.uint8(y * 255)
		return y

	def setup(self):
		if not self.model is None:
			return
		model = DispResNet(self.resNetLayers, False).to(device)
		weights = torch.load(self.weightsFile, map_location=device)
		model.load_state_dict(weights['state_dict'])
		model.eval()
		self.model = model
