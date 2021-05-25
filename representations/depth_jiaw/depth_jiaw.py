import numpy as np
import torch
from skimage.transform import resize as imresize
from media_processing_lib.image import imgResize
from nwmodule.utilities import device
from matplotlib.cm import hot

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

	# dph = imgResize(dph, height=size[0], width=size[1])
	dph = imresize(dph, size)
	dph = (dph - scale[0]) / (scale[1] - scale[0])
	dph = np.clip(dph, 0, 1)

	return dph

def closest_fit(size, multiples):
	return [round(size[i] / multiples[i]) * multiples[i] for i in range(len(multiples))]


class DepthJiaw(Representation):
	def __init__(self, weightsFile:str, resNetLayers:int, trainHeight:int, trainWidth:int, minDepth:int, maxDepth:int):     # runHeight:None, runWidth:None
		model = DispResNet(resNetLayers, False).to(device)
		weights = torch.load(weightsFile, map_location=device)
		model.load_state_dict(weights['state_dict'])
		model.eval()
		self.model = model
		self.multiples = (64, 64)   # is it 32 tho?
		self.trainSize = (trainHeight, trainWidth)
		# self.runSize = (runHeight, runWidth)
		self.scale = (minDepth, maxDepth)

	def make(self, x):
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
		pass

