# pylint: disable=all
import numpy as np


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


def depth_to_pointcloud(depth, normalized_coords):
    return depth[:, :, None] * normalized_coords


def depth_to_normals(depth, sampling_grid, normalized_grid, max_dist, min_valid=1):
    H, W = depth.shape[:2]

    point_cloud = depth_to_pointcloud(depth, normalized_grid)

    windows_3D = point_cloud[sampling_grid[0], sampling_grid[1]]

    invalid_samples = sampling_grid[2]
    if max_dist is not None and max_dist >= 0:
        distance_to_center = np.linalg.norm(windows_3D - point_cloud.reshape(H, W, 1, 3), axis=-1)
        too_far = distance_to_center > max_dist
        invalid_samples = np.logical_or(invalid_samples, too_far)

    valid_pixels = None
    if min_valid is not None and min_valid > 0:
        valid_count = np.count_nonzero(~invalid_samples, axis=2)
        valid_pixels = valid_count >= min_valid
        invalid_samples[~valid_pixels] = True

    valid_count = np.count_nonzero(~invalid_samples, axis=2)
    windows_3D[invalid_samples] = 0

    # compute feature from 3D windows
    window_sum = np.sum(windows_3D, axis=2, keepdims=True)
    centroid = np.full_like(window_sum, np.nan)

    if valid_pixels is not None:
        centroid[valid_pixels] = window_sum[valid_pixels] / valid_count.reshape((H, W, 1, 1))[valid_pixels]
        windows_3D[valid_pixels] = windows_3D[valid_pixels] - centroid[valid_pixels]
    else:
        centroid = window_sum / valid_count.reshape((H, W, 1, 1))
        windows_3D = windows_3D - centroid

    windows_3D[invalid_samples] = 0

    covariance = np.transpose(windows_3D, (0, 1, 3, 2)) @ windows_3D
    u, s, vh = np.linalg.svd(covariance)
    normals = vh[:, :, -1]

    angle = np.sum(normalized_grid * normals, axis=-1)
    neg_angle = np.sum(normalized_grid * (-normals), axis=-1)
    normals = np.where((angle > neg_angle)[:, :, None], normals, -normals)

    return normals
