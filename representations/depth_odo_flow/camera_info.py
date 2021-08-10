from pathlib import Path

import numpy as np
from transforms3d import axangles

from .cam import fov_diag_to_intrinsic
from .utils_geometry import rpy_to_R, invert_pose, skew, compose_pose


class CameraInfo:
    def __init__(self, velocity_path, velocities_type=None, poses_type=None, dt=None, camera_params=None,
                 frame_resolution=None, ang_vel_path=None):
        data = np.load(velocity_path)
        self.velocities_type = velocities_type
        if self.velocities_type is not None:
            self.linear_velocity, self.angular_velocity = get_velocities_archive(data, velocities_type)
            if ang_vel_path is not None:
                ang_vel_path = Path(ang_vel_path) / Path(velocity_path).name
                if Path(ang_vel_path).exists():
                    d = np.load(ang_vel_path)
                    inds, ang_vels = d["frame_inds"], d["ang_vel"]
                    self.angular_velocity[inds] = ang_vels
                else:
                    assert False, "Angular velocity path does not exist"
        self.relative_poses_type = poses_type
        self.dt = dt
        self.camera_params = camera_params
        self._frame_resolution = frame_resolution
        self._K = None
        if "position_w_full" in data and "orientation_rpy_rad_gt" in data:
            ts_gt = data["position_w_full"]
            rpy_gt = data["orientation_rpy_rad_gt"]
            Rs_gt = [rpy_to_R(*rpy) for rpy in rpy_gt]

            my_local_to_corke_local = np.array([[0., 0., 1.],
                                                [-1., 0., 0.],
                                                [0., -1., 0.]])
            Rs_gt = [r @ my_local_to_corke_local for r in Rs_gt]
            self.gt_poses = [(R, t) for R, t in zip(Rs_gt, ts_gt)]

    @property
    def number_of_frames(self):
        return len(self.gt_poses)

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
            assert self.dt is not None, "Set dt of video"
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


def get_velocities_archive(velocities, velocities_type):
    if velocities_type == "gt_rot":
        tag_linear_velocity = "linear_velocity_gt_R_c"
        tag_angular_velocity = "angular_velocity_gt_R_c"
    elif velocities_type == "gt_direct":
        tag_linear_velocity = "linear_velocity_gt_d"
        tag_angular_velocity = "angular_velocity_gt_d"
    elif velocities_type == "gt_direct_rw":
        tag_linear_velocity = "linear_velocity_camera_gt_d_rw"
        tag_angular_velocity = "ang_velocity_camera_gt_d_rw"
    elif velocities_type == "gt_direct_rw_avgv":
        tag_linear_velocity = "linear_velocity_camera_gt_d_rw_avgv"
        tag_angular_velocity = "ang_velocity_camera_gt_d_rw_avgv"
    elif velocities_type == "gt_direct_rspl":
        tag_linear_velocity = "linear_velocity_camera_gt_d_rspl"
        tag_angular_velocity = "ang_velocity_camera_gt_d_rspl"
    else:
        if velocities_type in ("gt_lin_vel", "gt_lin_ang_vel"):
            tag_linear_velocity = "linear_velocity_gt_c"
        else:
            tag_linear_velocity = "linear_velocity_c"
        if velocities_type in ("gt_ang_vel", "gt_lin_ang_vel"):
            tag_angular_velocity = "angular_velocity_gt_c"
        else:
            tag_angular_velocity = "angular_velocity_c"
    linear_velocity = velocities[tag_linear_velocity]
    angular_velocity = velocities[tag_angular_velocity]
    return linear_velocity, angular_velocity