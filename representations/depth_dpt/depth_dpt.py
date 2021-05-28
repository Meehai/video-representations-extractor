import numpy as np
import torch
from skimage.transform import resize as imresize
from media_processing_lib.image import imgResize
from media_processing_lib.video import MPLVideo
from nwmodule.utilities import device
from matplotlib.cm import hot
from typing import Dict

from .dpt_depth import DPTDepthModel
from .transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2

from ..representation import Representation


def closest_fit(size, multiples):
	return [round(size[i] / multiples[i]) * multiples[i] for i in range(len(multiples))]


class DepthDpt(Representation):
	def __init__(self, weightsFile:str, trainHeight:int, trainWidth:int):
		model = DPTDepthModel(
			path=weightsFile,
			backbone="vitl16_384",
			non_negative=True,
		)
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
		model.eval()
		model.to(device)
		self.trainSize = (trainHeight, trainWidth)
		self.model = model
		self.originalScaling = False


	def make(self, video:MPLVideo, t:int, depenedencyInputs:Dict[str, np.ndarray]) -> np.ndarray:
		x = video[t]
		img_input = self.transform({"image": x / 255.})["image"]
		# print('tile shape postproc', img_input.shape)
		# compute
		with torch.no_grad():
			sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
			prediction = self.model.forward(sample)
			prediction = (
				torch.nn.functional.interpolate(
					prediction.unsqueeze(1),
					size=x.shape[:2],
					mode="bicubic",
					align_corners=False,
				)
					.squeeze()
					.cpu()
					.numpy()
			)

			depth_min = prediction.min()
			depth_max = prediction.max()
			prediction = (prediction - depth_min) / (depth_max - depth_min)
		return prediction

	def makeImage(self, x):
		y = hot(x)[..., 0:3]
		y = np.uint8(y * 255)
		return y

	def setup(self):
		pass

