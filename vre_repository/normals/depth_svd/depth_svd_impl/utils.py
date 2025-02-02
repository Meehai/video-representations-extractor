# pylint: disable=all
import numpy as np
from .cam import fov_diag_to_intrinsic

_GRID_CACHE = {}

def get_sampling_grid(width, height, window_size, stride):
    window_x = np.arange(0, stride * window_size, stride) - window_size // 2 * stride
    window_y = window_x.copy()
    window_y, window_x = np.meshgrid(window_y, window_x, indexing='ij')
    window_x = window_x.flatten()
    window_y = window_y.flatten()
    xs = np.arange(width)
    ys = np.arange(height)
    ys, xs = np.meshgrid(ys, xs, indexing='ij')
    xs_windows = xs[:, :, None] + window_x[None, None, :]
    ys_windows = ys[:, :, None] + window_y[None, None, :]
    xs_windows[xs_windows >= width] = -1
    ys_windows[ys_windows >= height] = -1
    invalid = np.logical_or(xs_windows < 0, ys_windows < 0)
    xs_windows[invalid] = 0
    ys_windows[invalid] = 0
    window_coords = np.stack((window_y, window_x), axis=1)
    window_pixel_dist = np.linalg.norm(window_coords, axis=1)
    return ys_windows, xs_windows, invalid, window_pixel_dist


def get_normalized_coords(width, height, K):
    us = np.arange(width)
    vs = np.arange(height)
    vs, us = np.meshgrid(vs, us, indexing='ij')
    fx, fy, u0, v0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (us - u0) / fx
    y = (vs - v0) / fy
    z = np.ones_like(x)
    return np.stack((x, y, z), axis=2)

def _get_grid(depth: np.ndarray, sensor_fov: int, window_size: int, stride: int,
              sensor_size: tuple[int, int], input_downsample_step: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape[:2]
    if (height, width) in _GRID_CACHE:
        return _GRID_CACHE[(height, width)]
    if input_downsample_step is not None:
        depth = depth[:: input_downsample_step, :: input_downsample_step]
    depth_height, depth_width = depth.shape[:2]
    sampling_grid = get_sampling_grid(depth_width, depth_height, window_size, stride)
    K = fov_diag_to_intrinsic(sensor_fov, (sensor_size[0], sensor_size[1]), (depth_width, depth_height))
    normalized_grid = get_normalized_coords(depth_width, depth_height, K)
    _GRID_CACHE[(height, width)] = sampling_grid, normalized_grid
    return sampling_grid, normalized_grid


def depth_to_normals(depth: np.ndarray, sensor_fov: int, window_size: int, stride: int,
                     sensor_size: tuple[int, int], input_downsample_step: int) -> np.ndarray:
    H, W = depth.shape[:2]
    sampling_grid, normalized_grid = _get_grid(depth, sensor_fov, window_size, stride,
                                               sensor_size, input_downsample_step)
    from contexttimer import Timer

    point_cloud = depth[:, :, None] * normalized_grid # depth_to_pointcloud(depth, normalized_coords=normalized_grid)

    with Timer(prefix="windows_3D"):
        windows_3D = point_cloud[sampling_grid[0], sampling_grid[1]]

    invalid_samples = sampling_grid[2]

    with Timer(prefix="valid_count"):
        valid_count = np.count_nonzero(~invalid_samples, axis=2)
    with Timer(prefix="windows_3D[invalid_samples] = 0"):
        windows_3D[invalid_samples] = 0

    # compute feature from 3D windows
    with Timer(prefix="window_sum"):
        window_sum = np.sum(windows_3D, axis=2, keepdims=True)

    with Timer(prefix="centroid"):
        centroid = window_sum / valid_count.reshape((H, W, 1, 1))
        windows_3D = windows_3D - centroid

    with Timer(prefix="windows_3D[invalid_samples] = 0"):
        windows_3D[invalid_samples] = 0

    with Timer(prefix="covariance"):
        covariance = np.transpose(windows_3D, (0, 1, 3, 2)) @ windows_3D
    with Timer(prefix="svd"):
        u, s, vh = np.linalg.svd(covariance)
        normals = vh[:, :, -1]

    angle = np.sum(normalized_grid * normals, axis=-1)
    neg_angle = np.sum(normalized_grid * (-normals), axis=-1)
    normals = np.where((angle > neg_angle)[:, :, None], normals, -normals)

    return normals
