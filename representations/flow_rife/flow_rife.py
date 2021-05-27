import numpy as np
import torch
import torch.nn.functional as F
import flow_vis
from torchvision import transforms
from nwmodule.utilities import device
from media_processing_lib.image import imgResize

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

	def make(self, video, t):
		frame1 = video[t]
		frame2 = video[t + 1] if t < len(video) - 2 else frame1.copy()
		# frame1 = imgResize(frame1, width=256, height=256)
		# frame2 = imgResize(frame2, width=256, height=256)
		frame1 = (np.float32(frame1) / 255).transpose(2, 0, 1)[None]
		frame2 = (np.float32(frame2) / 255).transpose(2, 0, 1)[None]

		I0 = torch.from_numpy(frame1).to(device, non_blocking=True)
		I1 = torch.from_numpy(frame1).to(device, non_blocking=True)
		n, c, h, w = I0.shape
		ph = ((h - 1) // 32 + 1) * 32
		pw = ((w - 1) // 32 + 1) * 32
		padding = (0, pw - w, 0, ph - h)
		I0 = F.pad(I0, padding)
		I1 = F.pad(I1, padding)

		flow = self.model.inference(I0, I1)
		# flow = self.model.inference(I0, I1, self.UHD, self.no_backward_flow)
		flow = flow[0].cpu().numpy().transpose(1, 2, 0)
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

