import numpy as np
from pathlib import Path
from matplotlib.cm import hot
import os
from overrides import overrides

from .camera_info import CameraInfo, CameraSensorParams
from .depth_from_flow import depth_from_flow, filter_depth_from_flow
from ....representation import Representation, RepresentationOutput
from ....logger import logger
from ....utils import image_resize_batch


class DepthOdoFlow(Representation):
    def __init__(self, linear_ang_vel_correction: bool, focus_correction: bool,
                 cosine_correction_scipy: bool, cosine_correction_gd: bool, sensor_fov: int, sensor_width: int,
                 sensor_height: int, min_depth_meters: int, max_depth_meters: int, **kwargs):
        super().__init__(**kwargs)
        self.camera_params = CameraSensorParams(sensor_fov, (sensor_width, sensor_height))
        self.camera_info = CameraInfo(data=np.random.randn(len(self.video), 6), camera_params=self.camera_params)
        self._setup()
        self.linear_ang_vel_correction = linear_ang_vel_correction
        self.focus_correction = focus_correction
        self.cosine_correction_scipy = cosine_correction_scipy
        self.cosine_correction_gd = cosine_correction_gd
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
        assert len(self.dependencies) == 1, "Expected one optical flow method!"
        self.flow = self.dependencies[0]

    @overrides(check_signature=False)
    def vre_setup(self, velocities_path: str):
        velocities_path_pth = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/{velocities_path}").absolute()
        logger.info(f"Loading velocities from '{velocities_path_pth}'")
        data = np.load(velocities_path_pth)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = data["arr_0"]
        self.camera_info = CameraInfo(data=data, camera_params=self.camera_params)
        self.camera_info.dt = 1.0 / self.fps

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
                                                                       self.linear_ang_vel_correction,
                                                                       self.focus_correction,
                                                                       self.cosine_correction_gd,
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
    def make_images(self, t: slice, x: np.ndarray, extra: dict | None) -> np.ndarray:
        assert x.min() >= 0 and x.max() <= 1, (x.min(), x.max())
        x_rsz = image_resize_batch(x, height=self.video.frame_shape[0], width=self.video.frame_shape[1])
        where_max = np.where((x_rsz - 1).__abs__() < 1e-3)
        y = hot(x_rsz)[..., 0:3]
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
