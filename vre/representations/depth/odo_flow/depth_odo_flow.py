"""Depth from optical flow and odometry."""
from pathlib import Path
import os
import numpy as np
from matplotlib.cm import hot # pylint: disable=no-name-in-module
from overrides import overrides

from .odo_flow_impl.camera_info import CameraInfo, CameraSensorParams
from .odo_flow_impl.depth_from_flow import depth_from_flow, filter_depth_from_flow
from ....representation import Representation, RepresentationOutput
from ....logger import logger
from ....utils import image_resize_batch, VREVideo


class DepthOdoFlow(Representation):
    """Depth from optical flow and odometry."""
    def __init__(self, linear_ang_vel_correction: bool, focus_correction: bool,
                 cosine_correction_scipy: bool, cosine_correction_gd: bool, sensor_fov: int, sensor_width: int,
                 sensor_height: int, min_depth_meters: int, max_depth_meters: int, **kwargs):
        super().__init__(**kwargs)

        self.linear_ang_vel_correction = linear_ang_vel_correction
        self.focus_correction = focus_correction
        self.cosine_correction_scipy = cosine_correction_scipy
        self.cosine_correction_gd = cosine_correction_gd
        self.min_depth_meters = min_depth_meters
        self.max_depth_meters = max_depth_meters
        # TODO: thresholds picked for flow at 960x540; scaled correspondingly in filter function
        self.thresholds = {
            "Z": 0,
            ("Z", "around_focus_expansion_A"): 20,
            "angle (deg)": 20,
            "optical flow norm (pixels/s)": 20,
            "A norm (pixels*m/s)": 1,
        }
        assert len(self.dependencies) == 1, "Expected one optical flow method!"
        self.flow = self.dependencies[0]

        self.camera_params = CameraSensorParams(sensor_fov, (sensor_width, sensor_height))
        # dummy camera info so we can run with vre_setup() in tests.
        self.camera_info = CameraInfo(data=np.random.randn(100, 6, 540, 960), dt=1, camera_params=self.camera_params)
        self._linear_velocities: np.ndarray | None = None
        self._angular_velocities: np.ndarray | None = None

    # pylint: disable=arguments-differ
    @overrides(check_signature=False)
    def vre_setup(self, video: VREVideo, velocities_path: str):
        velocities_path_pth = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/{velocities_path}").absolute()
        logger.info(f"Loading velocities from '{velocities_path_pth}'")
        data = np.load(velocities_path_pth)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = data["arr_0"]
        self.camera_info = CameraInfo(data=data, dt=1 / video.frame_rate, camera_params=self.camera_params)
        self._linear_velocities = data[:, 0:3]
        self._angular_velocities = data[:, 3:6]

    @overrides
    def vre_dep_data(self, video: VREVideo, ix: slice) -> dict[str, RepresentationOutput]:
        flows = self.flow.vre_make(video, ix, make_images=False)[0][0]
        lin_vel = self._linear_velocities[ix]
        ang_vel = self._angular_velocities[ix]
        return {"flows": flows, "lin_vel": lin_vel, "ang_vel": ang_vel}

    # pylint: disable=arguments-differ
    @overrides(check_signature=False)
    def make(self, frames: np.ndarray, lin_vel: np.ndarray, ang_vel: np.ndarray,
             flows: np.ndarray) -> RepresentationOutput:
        # [0:1] -> [-1:1]
        flows = flows * 2 - 1
        # [-1:1] -> [-px:px]
        flow_height, flow_width = flows.shape[1:3]
        flows = flows * [flow_height, flow_width]
        # (B, H, W, 2) -> (B, 2, H, W)
        flows = flows.transpose(0, 3, 1, 2)

        if not self.camera_info.has_K():
            self.camera_info.frame_resolution = (flow_width, flow_height)
        flows = flows / self.camera_info.dt

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
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        x = repr_data[0]
        assert x.min() >= 0 and x.max() <= 1, (x.min(), x.max())
        x_rsz = image_resize_batch(x, height=frames.shape[1], width=frames.shape[2])
        where_max = np.where(np.abs(x_rsz - 1) < 1e-3)
        y = hot(x_rsz)[..., 0:3]
        y = np.uint8(y * 255)
        y[where_max] = [0, 0, 0]
        return y
