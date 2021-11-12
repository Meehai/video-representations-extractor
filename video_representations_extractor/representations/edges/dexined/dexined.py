import os
import gdown
import numpy as np
import torch as tr
from shutil import copyfile
from pathlib import Path
from nwutils.nwmodule import trModuleWrapper
from media_processing_lib.image import imgResize
from media_processing_lib.video import MPLVideo
from scipy.special import expit as sigmoid
from typing import Dict

from .model_dexined import DexiNed as Model
from ...representation import Representation
from ....logger import logger

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

def preprocessImage(img):
	img, coordinates = imgResize(img, height=352, width=352, mode="black_bars", \
		resizeLib="lycon", returnCoordinates=True)
	img = np.float32(img) / 255
	img = img.transpose(2, 0, 1)[None]
	return img, coordinates

def postprocessImage(img, coordinates):
	img = np.array(img)
	img = sigmoid(img)
	img[img < 0.0] = 0.0
	img = 1 - img
	img = img.mean(axis=0)[0][0]
	x0, y0, x1, y1 = coordinates
	img = img[y0 : y1, x0 : x1]
	return img

class DexiNed(Representation):
	def __init__(self, name, dependencies, saveResults:str, dependencyAliases):
		super().__init__(name, dependencies, saveResults, dependencyAliases)
		self.model = None
		self.weightsFile = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/deined.pth").absolute()

	def setup(self):
		# original files
		# urlWeights = "https://drive.google.com/u/0/uc?id=1MRUlg_mRwDiBiQLKFVuEfuvkzs65JFVe"
		# our backup
		urlWeights = "https://drive.google.com/u/0/uc?id=1oT1iKdRRKJpQO-DTYWUnZSK51QnJ-mnP"

		if not self.weightsFile.exists():
			logger.debug(f"Downloading weights for dexined from {urlWeights}")
			gdown.download(urlWeights, str(self.weightsFile))

		if self.model is None:
			model = Model().to(device)
			model.load_state_dict(tr.load(self.weightsFile, map_location=device))
			self.model = trModuleWrapper(model)

	def make(self, t:int) -> np.ndarray:
		A, coordinates = preprocessImage(self.video[t])
		with tr.no_grad():
			B = self.model.npForward(A)
		C = postprocessImage(B, coordinates)
		return C

	def makeImage(self, x):
		x = np.repeat(np.expand_dims(x["data"], axis=-1), 3, axis=-1)
		return np.uint8(x * 255)
