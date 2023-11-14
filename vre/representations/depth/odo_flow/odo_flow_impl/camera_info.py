# pylint: disable=all
import numpy as np
from transforms3d import axangles

from .cam import fov_diag_to_intrinsic
from .utils_geometry import invert_pose, skew, compose_pose

class CameraInfo:
    def __init__(self, data: np.ndarray, dt: float, poses_type=None, camera_params=None, frame_resolution=None):
        assert data.shape[1] == 6, f"Velocities data must have 6 values (3 linear, 3 angular). Got {data.shape[1]}."
        # data::(N, 6). Linear velocity is stored in first 3 components, angular in last 3.
        self.linear_velocity, self.angular_velocity = data[:, 0:3], data[:, 3:]
        self.relative_poses_type = poses_type
        self.dt = dt
        self.camera_params = camera_params
        self._frame_resolution = frame_resolution
        self._K = None

    @property
    def number_of_frames(self) -> int:
        return len(self.linear_velocity)

    @property
    def frame_resolution(self):
        return self._frame_resolution

    @frame_resolution.setter
    def frame_resolution(self, value):
        self._frame_resolution = value
        self._K = fov_diag_to_intrinsic(self.camera_params.fov, self.camera_params.sensor_resolution, value)

    def get_essential_matrix(self, frame_ind1, frame_ind2):
        delta_pose = self.get_delta_pose(frame_ind1, frame_ind2)
        R, t = invert_pose(*delta_pose)  # 2 to 1
        E = skew(t) @ R
        return E

    def get_fundamental_matrix(self, frame_ind1, frame_ind2):
        E = self.get_essential_matrix(frame_ind1, frame_ind2)
        invK = np.linalg.inv(self.K)
        F = invK.T @ E @ invK
        return F

    def get_projection_matrix(self, frame_ind1, frame_ind2):
        delta_pose = self.get_delta_pose(frame_ind1, frame_ind2)
        R, t = invert_pose(*delta_pose)  # 2 to 1
        P = self.K @ np.hstack((R, t[:, None]))
        return P

    def has_K(self):
        return self._K is not None

    @property
    def K(self):
        assert self._K is not None, "Must set K or frame resolution to initialize K"
        return self._K

    @K.setter
    def K(self, K):
        self._K = K

    def get_delta_pose(self, frame_ind1, frame_ind2):
        assert self.relative_poses_type is not None, "Set type of relative poses"
        if self.relative_poses_type == "integrated":
            assert self.velocities_type is not None, "Set type of velocities to use"
            return get_delta_pose(self.linear_velocity, self.angular_velocity, self.K, self.dt, frame_ind1, frame_ind2)
        elif self.relative_poses_type == "gt":
            delta_pose = compose_pose(invert_pose(*self.gt_poses[frame_ind1]), self.gt_poses[frame_ind2])
            return delta_pose


class CameraSensorParams:
    def __init__(self, fov=None, sensor_resolution=(3840, 2160)):
        self.fov = fov
        self.sensor_resolution = sensor_resolution


def get_delta_pose(linear_velocity, angular_velocity, K, dt, frame_ind1, frame_ind2):
    acc_pose = None
    for ind in range(frame_ind1, frame_ind2):
        v = linear_velocity[ind]
        w = angular_velocity[ind]
        t = v * dt
        angle = np.linalg.norm(w) * dt
        if angle > 0:
            R = axangles.axangle2mat(w, angle)
        else:
            R = np.eye(3)
        if acc_pose is not None:
            acc_pose = compose_pose(acc_pose, (R, t))
        else:
            acc_pose = (R, t)
    return acc_pose
