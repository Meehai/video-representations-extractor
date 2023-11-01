import numpy as np
from matplotlib.cm import hot
from typing import List
from overrides import overrides

from .camera_info import CameraInfo, CameraSensorParams
from .depth_from_flow import depth_from_flow, filter_depth_from_flow
from ....representation import Representation, RepresentationOutput


class DepthOdoFlow(Representation):
    def __init__(self, velocities_path: str, linearAngVelCorrection: bool, focus_correction: bool,
                 cosine_correction_scipy: bool, cosine_correction_GD: bool, sensor_fov: int, sensor_width: int,
                 sensor_height: int, min_depth_meters: int, max_depth_meters: int, **kwargs):
        self.camera_info = CameraInfo(velocities_path,
                                      camera_params=CameraSensorParams(sensor_fov, (sensor_width, sensor_height)))
        self.linearAngVelCorrection = linearAngVelCorrection
        self.focus_correction = focus_correction
        self.cosine_correction_scipy = cosine_correction_scipy
        self.cosine_correction_GD = cosine_correction_GD
        self.min_depth_meters = min_depth_meters
        self.max_depth_meters = max_depth_meters
        # thresholds picked for flow at 960x540; scaled correspondingly in filter function
        self.thresholds = {
            "Z": 0,
            ("Z", "around_focus_expansion_A"): 20,
            "angle (deg)": 20,
            "optical flow norm (pixels/s)": 20,
            "A norm (pixels*m/s)": 1,
        }
        super().__init__(**kwargs)
        self._setup()
        assert len(self.dependencies) == 1, "Expected one optical flow method!"
        self.flow = self.dependencies[0]

    @overrides
    def make(self, t: slice) -> RepresentationOutput:
        # [0:1] -> [-1:1]
        flows, _ = self.flow[t]

        flows = flows * 2 - 1
        flow_height, flow_width = flows.shape[1:3]
        # [-1:1] -> [-px:px]
        flows = flows * [flow_height, flow_width]
        # (B, H, W, 2) -> (B, 2, H, W)
        flows = flows.transpose(0, 3, 1, 2)
        if not self.camera_info.has_K():
            self.camera_info.frame_resolution = (flow_width, flow_height)
        flows = flows / self.camera_info.dt

        lin_vel = self.camera_info.linear_velocity[t]
        ang_vel = self.camera_info.angular_velocity[t]

        Zs, As, bs, derotating_flows, batch_ang_velc = depth_from_flow(flows, lin_vel, ang_vel, self.camera_info.K,
                                                                       self.linearAngVelCorrection,
                                                                       self.focus_correction,
                                                                       self.cosine_correction_GD,
                                                                       self.cosine_correction_scipy)
        valid = filter_depth_from_flow(Zs, As, bs, derotating_flows, thresholds=self.thresholds)

        Zs[~valid] = np.nan
        depth = np.clip(Zs.astype(np.float32), self.min_depth_meters, self.max_depth_meters)
        depth = (depth - self.min_depth_meters) / (self.max_depth_meters - self.min_depth_meters)
        depth[~np.isfinite(depth)] = 1

        extra = [{"rangedScaled": (self.min_depth_meters, self.max_depth_meters), "rangedValid": (0, 1),
                  "corrected_angular_velocity": batch_ang_velc[i]}
                 for i in range(len(batch_ang_velc))]

        return depth, extra

    @overrides
    def make_images(self, x: np.ndarray, extra: dict | None) -> np.ndarray:
        where_max = np.where(x == 1)
        assert x.min() >= 0 and x.max() <= 1, (x.min(), x.max())
        y = hot(x)[..., 0:3]
        y = np.uint8(y * 255)
        y[where_max] = [0, 0, 0]
        return y

    def _setup(self):
        assert len(self.camera_info.linear_velocity) == len(self.video), \
            f"{self.camera_info.linear_velocity.shape} vs {len(self.video)}"
        assert len(self.camera_info.angular_velocity) == len(self.video), \
            f"{self.camera_info.angular_velocity.shape} vs {len(self.video)}"
        self.fps = self.video.frame_rate
        self.camera_info.dt = 1.0 / self.fps
