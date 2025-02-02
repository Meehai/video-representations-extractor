# pylint: disable=all
import numpy as np
import math

_GRID_CACHE = {}

def fov_diag_to_intrinsic(fov, res, res_new=None):
    """
    Ideal camera matrix for diagonal fov and image size.
    :param fov: Diagonal FOV in degrees
    :param res: (original_width, original_height) tuple
    :param res_new: (rescaled_width, rescaled_height) tuple, optional
    :return: Camera matrix
    """
    d = math.sqrt((res[0] / 2) ** 2 + (res[1] / 2) ** 2)
    z = d / math.tan(fov * math.pi / 180 / 2)
    if res_new is None:
        k = np.asarray([[z, 0, 0], [0, z, 0], [res[0] / 2, res[1] / 2, 1]])
    else:
        ax = math.atan(res[0] / 2 / z)
        ay = math.atan(res[1] / 2 / z)
        fx = res_new[0] / 2 / math.tan(ax)
        fy = res_new[1] / 2 / math.tan(ay)
        k = np.asarray([[fx, 0, 0], [0, fy, 0], [res_new[0] / 2, res_new[1] / 2, 1]])
    return k.transpose().astype(np.float32)

def get_sampling_grid(width: int, height: int, window_size: int, stride: int) -> np.ndarray:
    if (key := (width, height, window_size, stride)) in _GRID_CACHE:
        return _GRID_CACHE[key]
    window_x = np.arange(0, stride * window_size, stride, dtype=np.int32) - window_size // 2 * stride
    window_y = window_x.copy()
    window_y, window_x = np.meshgrid(window_y, window_x, indexing="ij")
    window_x = window_x.flatten()
    window_y = window_y.flatten()
    xs = np.arange(width, dtype=np.int32)
    ys = np.arange(height, dtype=np.int32)
    ys, xs = np.meshgrid(ys, xs, indexing="ij")
    xs_windows = xs[:, :, None] + window_x[None, None, :]
    ys_windows = ys[:, :, None] + window_y[None, None, :]
    xs_windows[xs_windows >= width] = -1
    ys_windows[ys_windows >= height] = -1
    invalid = np.logical_or(xs_windows < 0, ys_windows < 0)
    xs_windows[invalid] = 0
    ys_windows[invalid] = 0
    #window_coords = np.stack((window_y, window_x), axis=1)
    #window_pixel_dist = np.linalg.norm(window_coords, axis=1)
    _GRID_CACHE[key] = ys_windows, xs_windows, invalid#, window_pixel_dist
    return ys_windows, xs_windows, invalid#, window_pixel_dist

def get_normalized_coords(width, height, K):
    us = np.arange(width)
    vs = np.arange(height)
    vs, us = np.meshgrid(vs, us, indexing="ij")
    fx, fy, u0, v0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (us - u0) / fx
    y = (vs - v0) / fy
    z = np.ones_like(x)
    return np.stack((x, y, z), axis=2).astype(np.float32)

def depths_to_normals(depths: np.ndarray, sensor_fov: int, window_size: int, stride: int,
                      sensor_size: tuple[int, int], input_downsample_step: int) -> np.ndarray:
    H, W = depths.shape[1], depths.shape[2]
    H_down, W_down = H // input_downsample_step, W // input_downsample_step
    K = fov_diag_to_intrinsic(sensor_fov, (sensor_size[0], sensor_size[1]), (W_down, H_down))
    normalized_grid = get_normalized_coords(W_down, H_down, K)[None].repeat(len(depths), axis=0)
    sampling_grid = [x[None].repeat(len(depths), axis=0)
                     for x in get_sampling_grid(W_down, H_down, window_size, stride)]

    point_clouds = depths[..., None] * normalized_grid

    X = (sampling_grid[0] * 0) + np.arange(len(depths)).reshape(-1, 1, 1, 1)
    windows_3D = point_clouds[X, sampling_grid[0], sampling_grid[1]]

    invalid_samples = sampling_grid[2]

    valid_count = np.count_nonzero(~invalid_samples, axis=3)
    windows_3D[invalid_samples] = 0

    # compute feature from 3D windows
    window_sum = np.sum(windows_3D, axis=3, keepdims=True)

    centroid = window_sum / valid_count.reshape((len(depths), H, W, 1, 1))
    windows_3D = windows_3D - centroid

    windows_3D[invalid_samples] = 0

    covariance = np.transpose(windows_3D, (0, 1, 2, 4, 3)) @ windows_3D
    _, _, vh = np.linalg.svd(covariance)
    normals = vh[:, :, :, -1]

    angle = (normalized_grid * normals).sum(axis=-1)
    neg_angle = (normalized_grid * -normals).sum(axis=-1)
    normals = np.where((angle > neg_angle)[..., None], normals, -normals)
    return normals
