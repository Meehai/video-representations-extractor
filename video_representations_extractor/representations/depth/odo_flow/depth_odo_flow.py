import numpy as np
from matplotlib.cm import hot
from typing import List
from overrides import overrides

from .camera_info import CameraInfo, CameraSensorParams
from .depth_from_flow import depth_from_flow, filter_depth_from_flow
from ...representation import Representation


class DepthOdoFlow(Representation):
	def __init__(self, name, dependencies:List[Representation], saveResults:str, \
			dependencyAliases:List[str], velocitiesPath:str, velocitiesType:str,
			linearAngVelCorrection:bool, focusCorrection:bool, cosineCorrectionScipy:bool, cosineCorrectionGD:bool,
			fov:int, sensorWidth:int, sensorHeight:int, minDepthMeters:int, maxDepthMeters:int):
		super().__init__(name, dependencies, saveResults, dependencyAliases)
		self.camera_info = CameraInfo(velocitiesPath, velocitiesType,
									  camera_params=CameraSensorParams(fov, (sensorWidth, sensorHeight)))
		self.linearAngVelCorrection = linearAngVelCorrection
		self.focusCorrection = focusCorrection
		self.cosineCorrectionScipy = cosineCorrectionScipy
		self.cosineCorrectionGD = cosineCorrectionGD
		self.minDepthMeters = minDepthMeters
		self.maxDepthMeters = maxDepthMeters
		# thresholds picked for flow at 960x540; scaled correspondingly in filter function
		self.thresholds = {
			"Z": 0,
			('Z', 'around_focus_expansion_A'): 20,
			"angle (deg)": 20,
			"optical flow norm (pixels/s)": 20,
			"A norm (pixels*m/s)": 1,
		}
		assert len(dependencies) == 1, "Expected one optical flow method!"
		self.flow = dependencies[0]

	def make(self, t):
		# [0:1] -> [-1:1]
		if t + 1 < len(self.video):
			data = self.flow[t]["data"]
			inverse_flow = False
		else:
			rawData = self.flow.get(t, t - 1)
			data = self.flow.resizeRawData(rawData)
			inverse_flow = True
		flow = data * 2 - 1
		flowHeight, flowWidth = flow.shape[0 : 2]
		# [-1:1] -> [-px:px]
		flow = flow * [flowHeight, flowWidth]
		if inverse_flow:
			flow = - flow
		if not self.camera_info.has_K():
			self.camera_info.frame_resolution = (flowWidth, flowHeight)
		flow = flow / self.camera_info.dt

		batched_flow = np.expand_dims(flow, axis=0).transpose(0, 3, 1, 2)
		batch_lin_vel = np.expand_dims(self.camera_info.linear_velocity[t], axis=0)
		batch_ang_vel = np.expand_dims(self.camera_info.angular_velocity[t], axis=0)

		Zs, As, bs, derotating_flows, batch_ang_velc = \
			depth_from_flow(batched_flow, batch_lin_vel, batch_ang_vel, self.camera_info.K,
							self.linearAngVelCorrection, self.focusCorrection, self.cosineCorrectionGD,
							self.cosineCorrectionScipy)
		valid = filter_depth_from_flow(Zs, As, bs, derotating_flows, thresholds=self.thresholds)

		Zs[~valid] = np.nan
		Zs = Zs[0]
		depth = np.clip(Zs.astype(np.float32), self.minDepthMeters, self.maxDepthMeters)
		depth = (depth - self.minDepthMeters) / (self.maxDepthMeters - self.minDepthMeters)
		depth[~np.isfinite(depth)] = 1
		return {"data": depth,
				"extra": {
					"rangeScaled": (self.minDepthMeters, self.maxDepthMeters),
					"rangeValid": (0, 1),
					"corrected_angular_velocity": batch_ang_velc[0],
					}
				}

	def makeImage(self, x):
		Where = np.where(x["data"] == 1)
		y = x["data"]
		assert y.min() >= 0 and y.max() <= 1
		y = hot(y)[..., 0:3]
		y = np.uint8(y * 255)
		y[Where] = [0, 0, 0]
		return y

	def setup(self):
		assert len(self.camera_info.linear_velocity) == len(self.video), "%d vs %d" % \
			(self.camera_info.linear_velocity.shape, len(self.video))
		assert len(self.camera_info.angular_velocity) == len(self.video), "%d vs %d" % \
			(self.camera_info.angular_velocity.shape, len(self.video))
		self.fps = self.video.fps
		self.camera_info.dt = 1. / self.fps
