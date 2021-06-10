import numpy as np
from matplotlib.cm import hot

from .cam import fov_diag_to_intrinsic
from .utils import depth_from_flow, filter_depth_from_flow
from ..representation import Representation

class DepthOdoFlow(Representation):
	def __init__(self, baseDir, name, dependencies, video, outShape, velocitiesPath:str, velocitiesType:str, \
		depth_axis:str, flowDirection:str, correct_ang_vel:bool, half_size:bool, fov:int):
		super().__init__(baseDir, name, dependencies, video, outShape)
		self.velocities = np.load(velocitiesPath)
		self.velocities_type = velocitiesType
		if self.velocities_type == 'gt_rot':
			tag_linear_velocity = 'linear_velocity_gt_R_c'
			tag_angular_velocity = 'angular_velocity_gt_R_c'
		elif self.velocities_type == 'gt_direct':
			tag_linear_velocity = 'linear_velocity_gt_d'
			tag_angular_velocity = 'angular_velocity_gt_d'
		else:
			if self.velocities_type in ('gt_lin_vel', 'gt_lin_ang_vel'):
				tag_linear_velocity = 'linear_velocity_gt_c'
			else:
				tag_linear_velocity = 'linear_velocity_c'
			if self.velocities_type in ('gt_ang_vel', 'gt_lin_ang_vel'):
				tag_angular_velocity = 'angular_velocity_gt_c'
			else:
				tag_angular_velocity = 'angular_velocity_c'
		self.linear_velocity = self.velocities[tag_linear_velocity]
		self.angular_velocity = self.velocities[tag_angular_velocity]
		assert len(self.linear_velocity) == len(self.video), "%d vs %d" % \
			(self.linear_velocity.shape, len(self.video))
		assert len(self.angular_velocity) == len(self.video), "%d vs %d" % \
			(self.angular_velocity.shape, len(self.video))
		self.depth_axis = depth_axis
		self.flowDirection = flowDirection
		self.correct_ang_vel = correct_ang_vel
		self.half_size = half_size
		self.fov = fov
		# thresholds picked for flow at 960x540; scaled correspondingly in filter function
		self.thresholds = {
			"Z": 0,
			('Z', 'around_focus_expansion_A'): 20,
			"angle (deg)": 20,
			"optical flow norm (pixels/s)": 20,
			"A norm (pixels*m/s)": 1,
		}
		self.fps = self.video.fps
		self.dt = 1. / self.fps
		assert len(dependencies) == 1, "Expected one optical flow method!"
		self.flow = dependencies[list(dependencies.keys())[0]]

	def make(self, t):
		# [0:1] -> [-1:1]
		flow = self.flow[t]["rawData"] * 2 - 1
		flowHeight, flowWidth = flow.shape[0 : 2]
		# [-1:1] -> [-px:px]
		flow = flow * [flowHeight, flowWidth]
		K = fov_diag_to_intrinsic(self.fov, (3840, 2160), (flowWidth, flowHeight))
		flow = flow / self.dt

		batched_flow = np.expand_dims(flow, axis=0).transpose(0, 3, 1, 2)
		batch_lin_vel = np.expand_dims(self.linear_velocity[t], axis=0)
		batch_ang_vel = np.expand_dims(self.angular_velocity[t], axis=0)

		Zs, As, bs, derotating_flows, batch_ang_velc = \
			depth_from_flow(batched_flow, batch_lin_vel, batch_ang_vel, K, self.depth_axis, self.correct_ang_vel)
		valid = filter_depth_from_flow(Zs, As, bs, derotating_flows, thresholds=self.thresholds)

		Zs[~valid] = np.nan
		Zs = Zs[0]
		depth_limits = (0, 400)
		depth = np.clip(Zs.astype(np.float32), depth_limits[0], depth_limits[1])
		depth = depth / depth_limits[1]
		depth[~np.isfinite(depth)] = 1
		return depth

	def makeImage(self, x):
		Where = np.where(x["data"] == 1)
		y = x["data"]
		assert y.min() >= 0 and y.max() <= 1
		y = hot(y)[..., 0:3]
		y = np.uint8(y * 255)
		y[Where] = [0, 0, 0]
		return y

	def setup(self):
		pass

