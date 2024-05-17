# pylint: disable=all
import pyproj
from scipy import integrate
from scipy.spatial.transform import Rotation, Slerp, RotationSpline
from scipy.interpolate import interp1d
import numpy as np
from transforms3d import quaternions as quat, euler, axangles

from .utils import nn_interpolation


def transform_points(camera_pose, points):
    if len(points) == 0:
        return points
    R, t = camera_pose
    points_out = (R @ points.T).T + t.reshape(1, -1)
    if len(points.shape) == 1:
        points_out = np.squeeze(points_out, axis=0)
    return points_out


def invert_pose(R, t):
    return R.T, (-R.T @ t.reshape(-1, 1)).flatten()


def compose_pose(p1, p2):
    try:
        R = p1[0] @ p2[0]
    except:
        pass
    t = (p1[0] @ p2[1].reshape(-1, 1)).flatten() + p1[1]
    return R, t


def pose_to_matrix(pose):
    T = np.zeros((4, 4))
    T[:3, :3] = pose[0]
    T[:3, 3] = pose[1]
    return T

def get_plane_pose(plane):
    A, B, C, D = plane
    # two different points on the plane
    p1 = np.array((0, 0, -D / C))
    p2 = np.array((-D / A, 0, 0))

    # relative vectors defines a direction parallel to the plane
    ox = p2 - p1
    ox = ox / np.linalg.norm(ox)

    # normal of the plane defines the second direction
    oz = np.array((A, B, C))
    oz = oz / np.linalg.norm(oz)

    # their cross product defines the third direction, Oz x Ox is poitive
    oy = np.cross(oz, ox)
    oy = oy / np.linalg.norm(oy)

    R = np.stack((ox, oy, oz), axis=1)
    assert (np.allclose(np.eye(3), R @ R.T)), f"{R} is not orthogonal; RR^T={R @ R.T}"
    t = p1
    return R, t


def err_to_plane(points, eq):
    return points.dot(np.array(eq[:3])) + eq[3]


def align(pointset_source, pointset_target, disable_reflection=False, delta_rot=None):
    centered_target = pointset_target - np.mean(pointset_target, axis=0)
    centered_source = pointset_source - np.mean(pointset_source, axis=0)
    rot, t = kabsch_alg(centered_target, centered_source, disable_reflection)
    if delta_rot is not None:
        rot = rot @ delta_rot

    return centered_source, centered_target @ rot


def kabsch_alg(pointset_source, pointset_target, disable_reflection=True):
    center_source = np.mean(pointset_source, axis=0)
    center_target = np.mean(pointset_target, axis=0)
    centered_target = pointset_target - center_target
    centered_source = pointset_source - center_source
    rot = rotation_between_points(centered_source, centered_target, disable_reflection)
    t = - rot.dot(center_source) + center_target
    return rot, t


def rotation_between_points(centered_source, centered_target, disable_reflection=True):
    # Sorkine-Hornung, Olga, and Michael Rabinovich. "Least-squares rigid motion using svd." Computing 1.1 (2017): 1-5.
    # find rotation which rotates source into target R * source = target
    covariance = centered_source.T @ centered_target
    u, s, vh = np.linalg.svd(covariance)
    v = vh.T
    rot = v @ u.T
    det = np.linalg.det(rot)
    if det < 0 and disable_reflection:
        v[:, -1] = - v[:, -1]
        rot = v @ u.T
    return rot


def qmult_smallest_angle(q1, q2):
    ax1, angle1 = quat.quat2axangle(quat.qmult(q1, q2))
    ax2, angle2 = quat.quat2axangle(quat.qmult(q1, -q2))
    if angle1 < angle2:
        return ax1, angle1
    else:
        return ax2, angle2


def rpy_to_R(r, p, y):
    # return (R.from_euler('z', y) * R.from_euler('y', p) *  R.from_euler('x', r)).as_matrix()
    return euler.euler2mat(y, p, r, 'rzyx')


def R_to_rpy(R):
    return euler.mat2euler(R, 'rzyx')[::-1]


def clip_angle(angle, low=-np.pi, high=np.pi):
    angle = np.where(angle > high, low + (angle - high), angle)
    angle = np.where(angle < low, high - (low - angle), angle)
    return angle


def iterative_scaling(d1, d2, inlier_thres=10, max_iterations=1000, initial_range=(50, 150)):
    inliers = np.logical_and(
        np.logical_and(d1 >= initial_range[0], d1 <= initial_range[1]),
        np.logical_and(d2 >= initial_range[0], d2 <= initial_range[1]),
    )
    count = np.count_nonzero(inliers)
    past_inliers = [(inliers, count)]
    solution = None
    loop_start = None
    for i in range(max_iterations):
        s = np.median(d1[inliers]) / np.median(d2[inliers])
        scaled_d2 = s * d2
        new_inliers = np.abs(d1 - scaled_d2) <= inlier_thres
        new_count = np.count_nonzero(new_inliers)
        if count == new_count and np.all(inliers == new_inliers):
            solution = inliers
            break
        loop_start = None
        for ind, (past_sol, past_count) in enumerate(past_inliers):
            if past_count == new_count and np.all(new_inliers == past_sol):
                loop_start = ind
                break
        if loop_start is not None:
            break
        past_inliers.append((new_inliers, new_count))
        inliers, count = new_inliers, new_count
    if loop_start is not None:
        best = loop_start
        for ind2 in range(loop_start + 1, len(past_inliers)):
            if past_inliers[ind2][1] > past_inliers[best][1]:
                best = ind2
        solution = past_inliers[best][0]
    return np.median(d1[solution]) / np.median(d2[solution]), solution


def interp_rotations(source_t, Rs, target_t, method='slerp', max_degree=0):
    if method == 'slerp':
        assert (max_degree == 0)
        key_rots = Rotation.from_matrix(Rs)
        slerp = Slerp(source_t, key_rots)
        target_t_mask = np.logical_and(source_t[0] <= target_t, source_t[-1] >= target_t)
        interp_rots = slerp(target_t[target_t_mask])
        target_Rs = interp_rots.as_matrix()
        return nn_interpolation(target_t, target_Rs, target_t[target_t_mask])
    elif method == 'spline':
        key_rots = Rotation.from_matrix(Rs)
        spline = RotationSpline(source_t, key_rots)
        res = []
        target_t_mask = np.logical_and(source_t[0] <= target_t, source_t[-1] >= target_t)
        for deg in range(max_degree + 1):
            target = spline(target_t, deg)
            if deg == 0:
                target = target.as_matrix()
            res.append(nn_interpolation(target_t, target, target_t[target_t_mask]))
        res = res[0] if len(res) == 1 else res
        return res
    else:
        raise NotImplemented('Rotation interpolation method unknown.')


def interp_vectors(source_t, vectors, target_t):
    target_t_mask = np.logical_and(source_t[0] <= target_t, source_t[-1] >= target_t)
    interp_vectors = interp1d(source_t, vectors, axis=0)(target_t[target_t_mask])
    return nn_interpolation(target_t, interp_vectors, target_t[target_t_mask])


def diff_rot(orientations, ts, type='central'):
    angle_vels = []
    for ind in range(len(orientations)):
        prev_ind = next_ind = ind
        if type in ['central', 'left', 'right']:
            if type in ['central', 'left']:
                prev_ind = max(0, ind - 1)
            if type in ['central', 'right']:
                next_ind = min(ind + 1, len(orientations) - 1)
            delta_t = ts[next_ind] - ts[prev_ind]
            prev_R = orientations[prev_ind]
            next_R = orientations[next_ind]
            ax, angle = axangles.mat2axangle(prev_R.T.dot(next_R))
            if delta_t > 0:
                vel = ax * angle / delta_t
            else:
                vel = np.zeros(3)
            angle_vels.append(vel)
        elif type == 'mean':
            prev_ind = max(0, ind - 1)
            next_ind = min(ind + 1, len(orientations) - 1)
            R = orientations[ind]
            prev_R = orientations[prev_ind]
            next_R = orientations[next_ind]
            delta_forward = ts[next_ind] - ts[ind]
            delta_backward = ts[ind] - ts[prev_ind]

            ax_forward, angle_forward = axangles.mat2axangle(R.T.dot(next_R))
            ax_backward, angle_backward = axangles.mat2axangle(R.T.dot(prev_R))

            if not delta_forward > 0:
                ang_vel_backward = ax_backward * angle_backward / delta_backward
                mean_ang_vel = - ang_vel_backward
            elif not delta_backward > 0:
                ang_vel_forward = ax_forward * angle_forward / delta_forward
                mean_ang_vel = ang_vel_forward
            else:
                ang_vel_forward = ax_forward * angle_forward / delta_forward
                ang_vel_backward = ax_backward * angle_backward / delta_backward
                mean_ang_vel = (ang_vel_forward + (- ang_vel_backward)) / 2.
            angle_vels.append(mean_ang_vel)
        else:
            raise NotImplemented(f"Unknown rotation differentiation type: {type}")
    return np.array(angle_vels)


def geocentric_to_xyz(lat, lon, alt, radians=False):
    ecef = pyproj.CRS(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.CRS(proj='latlong', ellps='WGS84', datum='WGS84')
    transformer = pyproj.Transformer.from_crs(lla, ecef)
    x, y, z = transformer.transform(lon, lat, alt, radians=radians)
    return np.stack((x, y, z), axis=1)


def integrate_time_series(data, t, integration='trapezoidal'):
    if integration == 'trapezoidal':
        wrk = integrate.cumtrapz(data, x=t, axis=0, initial=0)
    elif integration == 'simpson':
        wrk = integrate.simps(data, x=t, axis=0)
    elif integration == 'rectangular':
        dt = np.zeros_like(t)
        dt[1:] = t[1:] - t[:-1]
        wrk = np.cumsum(data * dt.reshape((-1, 1)), axis=0)
    elif integration == 'midpoint':
        dt = t[1:] - t[:-1]
        avg = (data[1:] + data[:-1]) / 2
        wrk = np.zeros_like(data)
        wrk[1:] = np.cumsum(avg * dt.reshape((-1, 1)), axis=0)
    else:
        raise NotImplementedError(f'Not implemented: {integration}')

    return wrk


def integrate_angular_velocity(data, ts):
    prev_t = None
    norm = np.linalg.norm(data, axis=1)
    rotations = np.empty((len(data), 3, 3), dtype=np.float)
    rotations[0] = np.eye(3)
    for ind, (t, w) in enumerate(zip(ts, data)):
        if prev_t is not None:
            dt = t - prev_t
            angle = norm[ind]
            angle *= dt
            if angle != 0:
                ax = w / angle
            else:
                ax = np.ones(3)
            delta_R = axangles.axangle2mat(ax, angle * dt)
            rotations[ind] = rotations[ind - 1] @ delta_R
        prev_t = t
    return rotations


def get_grid(width, height, K=None):
    us = np.arange(width)
    vs = np.arange(height)
    vs, us = np.meshgrid(vs, us, indexing='ij')
    if K is None:
        return np.stack((us, vs), axis=2)
    fx, fy, u0, v0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (us - u0) / fx
    y = (vs - v0) / fy
    z = np.ones_like(x)
    return np.stack((us, vs), axis=2), np.stack((x, y, z), axis=2)


def skew(v):
    return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
    ])


def fundamental_matrix_error(p1, p2, F):
    a, b, c = F[:, :2] @ p1.T + F[:, 2:]
    d1 = np.abs(p2[:, 0] * a + p2[:, 1] * b + c) / np.sqrt(a ** 2 + b ** 2)
    a, b, c = F.T[:, :2] @ p2.T + F.T[:, 2:]
    d2 = np.abs(p1[:, 0] * a + p1[:, 1] * b + c) / np.sqrt(a ** 2 + b ** 2)

    return np.maximum(d1, d2)
