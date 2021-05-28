import numpy as np
import torch
import torch.nn.functional as F
import flow_vis
from typing import Dict
from torchvision import transforms
from nwmodule.utilities import device
from media_processing_lib.image import imgResize
from media_processing_lib.video import MPLVideo

from ..representation import Representation
from .RIFE_HDv2 import Model

class FlowRife(Representation):
	def __init__(self, weightsPath:str, computeBackwardFlow:bool):
		model = Model()
		model.load_model(weightsPath, -1)
		model.eval()
		model.device()
		torch.set_grad_enabled(False)
		self.model = model
		self.UHD = False
		self.no_backward_flow = True if computeBackwardFlow is None else not computeBackwardFlow

	def make(self, video:MPLVideo, t:int, depenedencyInputs:Dict[str, np.ndarray]) -> np.ndarray:
		frame1 = video[t]
		frame2 = video[t + 1] if t < len(video) - 2 else frame1.copy()
		
		# Convert, preprocess & pad
		I0 = torch.from_numpy(np.transpose(frame1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
		I1 = torch.from_numpy(np.transpose(frame2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
		n, c, h, w = I0.shape
		ph = ((h - 1) // 32 + 1) * 32
		pw = ((w - 1) // 32 + 1) * 32
		padding = (0, pw - w, 0, ph - h)
		I0 = F.pad(I0, padding)
		I1 = F.pad(I1, padding)

		flow = self.model.inference(I0, I1, self.UHD, self.no_backward_flow)

		# Convert, postprocess and remove pad
		flow = flow[0].cpu().numpy().transpose(1, 2, 0)
		returnedShape = flow.shape[0 : 2]
		# Remove the padding to keep original shape
		halfPh, halfPw = (ph - h) // 2, (pw - w) // 2
		flow = flow[0 : returnedShape[0]-halfPh, 0 : returnedShape[1]-halfPw]
		# [-px : px] => [-1 : 1]
		flow /= returnedShape
		# [-1 : 1] => [0 : 1]
		flow = (flow + 1) / 2
		return flow

	def makeImage(self, x):
		# [0 : 1] => [-1 : 1]
		x = x * 2 - 1
		y = flow_vis.flow_to_color(x)
		return y

	def setup(self):
		pass

