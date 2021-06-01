import numpy as np

from ..representation import Representation

from .cam import fov_diag_to_intrinsic
from pathlib import Path

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from .utils import map_timed, depth_from_flow, remove_padding, filter_depth_from_flow


class DepthOdoFlow(Representation):
	def __init__(self, baseDir, name, dependencies, video, outShape, velocitiesPath:str, batchSize:int, flowHeight:int,
				flowWidth:int, fps:int, velocitiesType:str):
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
		self.depth_axis = 'xy'
		self.flow = 'forward'
		self.correct_ang_vel = True
		self.half_size = True,
		# thresholds picked for flow at 960x540; scaled correspondingly in filter function
		self.thresholds = {
			"Z": 0,
			('Z', 'around_focus_expansion_A'): 20,
			"angle (deg)": 20,
			"optical flow norm (pixels/s)": 20,
			"A norm (pixels*m/s)": 1,
		}
		self.fov = 75
		self.fps = fps
		self.dt = 1. / self.fps
		self.K = fov_diag_to_intrinsic(self.fov, (3840, 2160), (flowWidth, flowHeight))
		self.batchSize = batchSize
		self.flow_pad = (0, 0, 0, 0)
		self.startFrame = 0
		self.depth_axis = 'xy'
		self.correct_ang_vel = True

		self.flow = dependencies['opticalflow1']



	def make(self, t):
		batch_inds = list(range(t, t+self.batchSize))
		# flows = [remove_padding(make_flow_inference(self.flowModel, x[ind], x[ind + 1]), self.flow_pad)
		# 		 for ind in range(len(batch_inds))]
		flows = [(self.flow[batch_inds[ind]]['data'] * 2 - 1) * self.flow[batch_inds[ind]]['data'].shape[0:2]
					for ind in range(len(batch_inds))]

		batched_flow = np.array(flows) / self.dt

		batched_flow = np.transpose(batched_flow, (0, 3, 1, 2))

		batch_lin_vel = self.linear_velocity[batch_inds]
		batch_ang_vel = self.angular_velocity[batch_inds]
		Zs, As, bs, derotating_flows, batch_ang_velc = depth_from_flow(batched_flow, batch_lin_vel, batch_ang_vel,
																	   self.K,
																	   self.depth_axis, self.correct_ang_vel)

		valid = filter_depth_from_flow(Zs, As, bs, derotating_flows, thresholds=self.thresholds)
		Zs[~valid] = np.nan
		Zs = Zs[0] # hopefully we won't waste batchSize -1 in the future
		depth_limits = (0, 400)
		depth = np.clip(Zs.astype(np.float32), depth_limits[0], depth_limits[1])
		depth = 1 - depth / depth_limits[1]

		return depth


	def makeImage(self, x):
		normalizer = mpl.colors.Normalize(vmin=0, vmax=1)
		cmap = plt.get_cmap("hot")
		cmap.set_bad(color=(0.0, 0.0, 0.0, 1.0))
		mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
		depth_to_image = lambda x: (mapper.to_rgba(x)[:, :, :3] * 255).astype(np.uint8)

		y = depth_to_image(x['data'])
		return y

	def setup(self):
		pass

