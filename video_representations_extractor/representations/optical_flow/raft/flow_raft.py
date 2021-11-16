import os
import numpy as np
import torch as tr
import flow_vis
import gdown
from overrides import overrides
from pathlib import Path
from media_processing_lib.image import imgResize

from .utils import InputPadder
from .raft import RAFT
from ...representation import Representation, RepresentationOutput
from ....logger import logger

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

class FlowRaft(Representation):
	def __init__(self, inputWidth:int, inputHeight:int, **kwargs):
		super().__init__(**kwargs)
		self.model = None
		self.weightsDir = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/raft")
		self.inputWidth = inputWidth
		self.inputHeight = inputHeight

		self.small = False
		self.mixed_precision = False

	@overrides
	def setup(self):
		# Pointless to upsample with bilinear, it's better we fix the video input.
		assert self.video.shape[1] >= self.inputHeight and self.video.shape[2] >= self.inputWidth, \
			f"{self.video.shape} vs {self.inputHeight}x{self.inputWidth}"
		self.weightsDir.mkdir(exist_ok=True)

		# original files
		raftThingsUrl = "https://drive.google.com/u/0/uc?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM"

		raftThingsPath = self.weightsDir / "raft-things.pkl"
		if not raftThingsPath.exists():
			logger.debug("Downloading weights for RAFT")
			gdown.download(raftThingsUrl, str(raftThingsPath))

		if self.model is None:
			model = tr.nn.DataParallel(RAFT(self))
			model.load_state_dict(tr.load(raftThingsPath, map_location=device))

			model = model.module
			model.to(device)
			model.eval()

			self.model = model

	@overrides
	def make(self, t: int) -> RepresentationOutput:
		t_target = t + 1 if t < len(self.video) - 1 else t
		return self.get(t, t_target)

	def get(self, t_source, t_target) -> np.ndarray:
		self.setup()

		frame1 = self.video[t_source]
		frame2 = self.video[t_target]

		frame1 = imgResize(frame1, height=self.inputHeight, width=self.inputWidth, interpolation="bilinear")
		frame2 = imgResize(frame2, height=self.inputHeight, width=self.inputWidth, interpolation="bilinear")

		# Convert, preprocess & pad
		frame1 = tr.from_numpy(np.transpose(frame1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float()
		frame2 = tr.from_numpy(np.transpose(frame2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float()

		padder = InputPadder(frame1.shape)
		image1, image2 = padder.pad(frame1, frame2)

		with tr.no_grad():
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

	@overrides
	def makeImage(self, x: RepresentationOutput) -> np.ndarray:
		# [0 : 1] => [-1 : 1]
		x = x["data"] * 2 - 1
		y = flow_vis.flow_to_color(x)
		return y
