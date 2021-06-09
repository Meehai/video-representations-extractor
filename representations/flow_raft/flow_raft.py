import numpy as np
import torch
import torch.nn.functional as F
import flow_vis
import gdown
from typing import Dict
from torchvision import transforms
from nwmodule.utilities import device
from nwdata.utils import fullPath
from media_processing_lib.image import imgResize
from media_processing_lib.video import MPLVideo

from ..representation import Representation
from .utils import InputPadder
from .raft import RAFT


class FlowRaft(Representation):
	def __init__(self, baseDir, name, dependencies, video, outShape, inputWidth:int, inputHeight:int):
		super().__init__(baseDir, name, dependencies, video, outShape)
		# Pointless to upsample with bilinear, it's better we fix the video input.
		assert video.shape[1] >= inputHeight and video.shape[2] >= inputWidth, "%s vs %dx%d" \
			% (video.shape, inputHeight, inputWidth)
		self.model = None
		self.weightsDir = fullPath(__file__).parents[2] / "weights/raft"
		self.inputWidth = inputWidth
		self.inputHeight = inputHeight

		self.output_downsample_step = 2
		self.small = False
		self.mixed_precision = False

	def setup(self):
		self.weightsDir.mkdir(exist_ok=True)

		# original files
		raftThingsUrl = "https://drive.google.com/u/0/uc?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM"

		raftThingsPath = self.weightsDir / "raft-things.pkl"
		if not raftThingsPath.exists():
			print("[FlowRaft::setup] Downloading weights for RAFT")
			gdown.download(raftThingsUrl, str(raftThingsPath))

		if self.model is None:
			model = torch.nn.DataParallel(RAFT(self))
			model.load_state_dict(torch.load(raftThingsPath, map_location=device))

			model = model.module
			model.to(device)
			model.eval()

			self.model = model

	def make(self, t:int) -> np.ndarray:
		frame1 = self.video[t]
		frame2 = self.video[t + 1] if t < len(self.video) - 2 else frame1.copy()

		frame1 = imgResize(frame1, height=self.inputHeight, width=self.inputWidth, interpolation="bilinear")
		frame2 = imgResize(frame2, height=self.inputHeight, width=self.inputWidth, interpolation="bilinear")

		# Convert, preprocess & pad
		frame1 = torch.from_numpy(np.transpose(frame1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float()
		frame2 = torch.from_numpy(np.transpose(frame2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float()

		padder = InputPadder(frame1.shape)
		image1, image2 = padder.pad(frame1, frame2)

		with torch.no_grad():
			_, flow = self.model(image1, image2, iters=20, test_mode=True)

		# Convert, postprocess and remove pad
		flow = flow[0].cpu().numpy().transpose(1, 2, 0)
		returnedShape = flow.shape[0 : 2]
		# Remove the padding to keep original shape
		flow = padder.unpad(flow)
		# [-px : px] => [-1 : 1]
		flow /= returnedShape
		# [-1 : 1] => [0 : 1]
		flow = (flow + 1) / 2
		return flow

	def makeImage(self, x):
		# [0 : 1] => [-1 : 1]
		x = x["data"] * 2 - 1
		y = flow_vis.flow_to_color(x)
		return y
