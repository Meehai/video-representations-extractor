import numpy as np
import torch
import torch.nn.functional as F
import cv2
import gdown
from skimage.transform import resize as imresize
from torchvision.transforms import Compose
from media_processing_lib.video import MPLVideo
from nwmodule.utilities import device
from nwdata.utils import fullPath
from matplotlib.cm import hot
from typing import Dict

from .dpt_depth import DPTDepthModel
from .transforms import Resize, NormalizeImage, PrepareForNet

from ..representation import Representation


def closest_fit(size, multiples):
	return [round(size[i] / multiples[i]) * multiples[i] for i in range(len(multiples))]


class DepthDpt(Representation):
	def __init__(self, name, dependencies, trainHeight, trainWidth):
		super().__init__(name, dependencies)
		net_w, net_h = 384, 384
		resize_mode = "minimal"
		normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		self.transform = Compose(
			[
				Resize(
					net_w,
					net_h,
					resize_target=None,
					keep_aspect_ratio=True,
					ensure_multiple_of=32,
					resize_method=resize_mode,
					image_interpolation_method=cv2.INTER_CUBIC,
				),
				normalization,
				PrepareForNet(),
			]
		)
		self.trainSize = (trainHeight, trainWidth)
		self.originalScaling = False
		self.model = None
		self.weightsFile = str(fullPath(__file__).parents[2] / "weights/depth_dpt_midas.pth")

	def setup(self):
		# our backup
		urlWeights = "https://drive.google.com/u/0/uc?id=15JbN2YSkZFSaSV2CGkU1kVSxCBrNtyhD"

		weightsPath = fullPath(self.weightsFile)
		if not weightsPath.exists():
			print("[DexiNed::setup] Downloading weights for dexined from %s" % urlWeights)
			gdown.download(urlWeights, self.weightsFile)

		if self.model is None:
			model = DPTDepthModel(
				path=self.weightsFile,
				backbone="vitl16_384",
				non_negative=True,
			)
			model.eval()
			model.to(device)
			self.model = model

	def make(self, t:int) -> np.ndarray:
		x = self.video[t]
		img_input = self.transform({"image": x / 255.})["image"]
		# compute
		with torch.no_grad():
			sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
			prediction = self.model.forward(sample).squeeze(dim=1)
			prediction = prediction.squeeze().cpu().numpy()
			prediction = 1 / prediction
			prediction = np.clip(prediction, 0, 1)
		return prediction

	def makeImage(self, x):
		y = x["data"]
		y = hot(y)[..., 0 : 3]
		y = np.uint8(y * 255)
		return y
