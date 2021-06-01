from pathlib import Path
import numpy as np
import multiprocessing
import cv2
import time
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm

import pandas as pd
from matplotlib import pyplot as plt

from .polyfit import linear_least_squares


def reduce_list(data, mask=None, indices=None):
    assert (mask is None or indices is None)
    if mask is not None:
        return [data[ind] for ind in range(len(mask)) if mask[ind]]
    if indices is not None:
        return [data[ind] for ind in indices]
    return data


def get_params_from_obj(x):
    return dict((key, getattr(x, key)) for key in dir(x) if key not in dir(x.__class__) and not key.startswith("__"))


def write_img(path, img):
    path = Path(path).with_suffix(".png")
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def read_img(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def is_sorted(ids):
    ids = np.array(ids)
    return np.all((ids[1:] - ids[:-1]) == 1)


def map_timed(func, iterable, size, parallel=True, max_cpu_count=4, print_step=100):
    start = time.time()
    results = []
    if parallel:
        cpu_count = min(max_cpu_count, multiprocessing.cpu_count())
        pool = multiprocessing.Pool(cpu_count)
        cpu_count = multiprocessing.cpu_count()
        chunk_size = min(100, size // cpu_count)
        count = 0
        for res in pool.imap(func, iterable, chunk_size):
            count += 1
            results.append(res)
            if count % print_step == 0:
                print(f"Processed {count}/{size}; {count / (time.time() - start):.2f} items/sec")
    else:
        count = 0
        for d in iterable:
            r = func(d)
            count += 1
            if count % print_step == 0:
                print(f"Processed {count}/{size}; {count / (time.time() - start):.2f} items/sec")
            results.append((r))
    return results


def solve_delta_ang_vel(jacobian_z, jacobian_ang_vel, b):
    # A is 2 x W x H
    _, W, H = jacobian_z.shape
    Ar = np.zeros((2 * W * H, W * H + 3))
    np.fill_diagonal(Ar[::2, :-3], jacobian_z[0].flatten())
    np.fill_diagonal(Ar[1::2, :-3], jacobian_z[1].flatten())
    Ar[:, -3:] = np.transpose(jacobian_ang_vel, (2, 3, 0, 1)).reshape(-1, 3)
    # Ar = np.zeros((2 * W * H, W * H ))
    # np.fill_diagonal(Ar[::2], jacobian_z[0].flatten())
    # np.fill_diagonal(Ar[1::2], jacobian_z[1].flatten())
    scale = np.max(np.abs(Ar))
    br = np.transpose(b, [1, 2, 0]).flatten()
    Ar = Ar / scale
    br = br / scale
    solution = linear_least_squares(Ar, br)
    return solution


def solve_delta_ang_vel_v2(jacobian_z, jacobian_ang_vel, b):
    # A is 2 x num_pixels
    _, num_pixels = jacobian_z.shape
    Ar = np.zeros((2 * num_pixels, num_pixels + 3))
    np.fill_diagonal(Ar[::2, :-3], jacobian_z[0].flatten())
    np.fill_diagonal(Ar[1::2, :-3], jacobian_z[1].flatten())
    Ar[:, -3:] = np.transpose(jacobian_ang_vel, (2, 0, 1)).reshape(-1, 3)
    # Ar = np.zeros((2 * W * H, W * H ))
    # np.fill_diagonal(Ar[::2], jacobian_z[0].flatten())
    # np.fill_diagonal(Ar[1::2], jacobian_z[1].flatten())
    scale = np.max(np.abs(Ar))
    br = np.transpose(b, [1, 0]).flatten()
    Ar = Ar / scale
    br = br / scale
    # if not positive_z:
    solution = linear_least_squares(Ar, br)
    # else:
    #     x = cp.Variable(shape=(Ar.shape[1], ))
    #     constraints = [x[:-3] >= 1/5000, x[:-3] <= 1/10]
    #     obj = cp.Minimize(cp.sum_squares(Ar @ x - br))
    #     prob = cp.Problem(obj, constraints)
    #     prob.solve()  # Returns the optimal value.
    #     solution = np.asarray(x.value)
    return solution


def depth_from_flow(batched_flow, linear_velocity, angular_velocity, K, axis=('x', 'y', 'xy')[0], adjust_ang_vel=True,
                    mesh_grid=None):
    f_u, f_v = K[0, 0], K[1, 1]
    u0, v0 = K[0, 2], K[1, 2]

    H, W = batched_flow.shape[2:]

    if mesh_grid is not None:
        us_bar, vs_bar = mesh_grid
    else:
        us_bar, vs_bar = init_mesh_grid(H, W, u0, v0)

    B = len(linear_velocity)

    A = get_A(linear_velocity, us_bar, vs_bar, f_u, f_v)

    us_bar_sq = us_bar ** 2
    vs_bar_sq = vs_bar ** 2
    uv = us_bar * vs_bar

    # derotating_flow = np.empty((B, 2, H, W))
    angular_velocity = np.expand_dims(angular_velocity.copy(), axis=(2, 3))

    J_w = np.array([[uv / f_v, - (f_u ** 2 + us_bar_sq) / f_u, f_u / f_v * vs_bar],
                    [(f_v ** 2 + vs_bar_sq) / f_v, - uv / f_u, - f_v / f_u * us_bar]])

    derotating_flow = get_derotating_flow(J_w, angular_velocity)

    # compute b
    b = batched_flow - derotating_flow
    if adjust_ang_vel:
        step = int(80 * H / 540) # for 12x7 grid at 540x960

        for b_ind in range(B):

            mask = np.ones_like(us_bar, dtype=bool)
            mask_sample = mask[::step, ::step]
            # mask_sample.fill(True)
            A_sample = A[b_ind][:, ::step, ::step][:, mask_sample]
            b_sample = b[b_ind][:, ::step, ::step][:, mask_sample]
            J_w_sample = J_w[:, :, ::step, ::step][:, :, mask_sample]
            solution = solve_delta_ang_vel_v2(A_sample, J_w_sample, b_sample)
            if solution is not None:
                ang_vel_correction = solution[-3:]
                angular_velocity[b_ind] = angular_velocity[b_ind] + ang_vel_correction.reshape(-1, 1, 1)
        derotating_flow = get_derotating_flow(J_w, angular_velocity)
        b = batched_flow - derotating_flow

    if axis == 'xy':
        # Z = norm(A)^2 / dot(A, b) (from least squares, A, b are 2x1 matrices, so vectors)
        norm_A_squared = np.sum(np.square(A), axis=1)
        dot_Ab = np.sum(A * b, axis=1)

        Z = norm_A_squared / dot_Ab
    elif axis == 'x':
        Z = A[:, 0] / b[:, 0]
    else:
        Z = A[:, 1] / b[:, 1]
    return Z, A, b, derotating_flow, np.squeeze(angular_velocity, axis=(2, 3))


def get_A(linear_velocity, us_bar, vs_bar, f_u, f_v):
    linear_velocity = np.expand_dims(linear_velocity, axis=(2, 3))
    linear_velocity = np.transpose(linear_velocity, [1, 0, 2, 3])
    A0 = - f_u * linear_velocity[0] + linear_velocity[2] * us_bar
    A1 = - f_v * linear_velocity[1] + linear_velocity[2] * vs_bar
    A = np.stack((A0, A1), axis=0)
    A = np.transpose(A, [1, 0, 2, 3])
    return A


def init_mesh_grid(H, W, u0, v0):
    us = np.arange(W)
    vs = np.arange(H)
    vs, us = np.meshgrid(vs, us, indexing='ij')
    us_bar = us - u0
    vs_bar = vs - v0
    return us_bar, vs_bar


def get_derotating_flow(J_w, angular_velocity):
    derotating_flow = np.transpose((
            np.transpose(J_w, (2, 3, 0, 1)) @ np.transpose(angular_velocity, (2, 3, 1, 0))),
        (3, 2, 0, 1))
    return derotating_flow


def filter_depth_from_flow(Zs, As, bs, derotating_flows, thresholds, virtual_height=540):
    valid = np.full_like(Zs, True, dtype=bool)
    for feature, threshold in thresholds.items():
        feature_data = get_feature_from_depth_from_flow_data(Zs, As, bs, derotating_flows, feature)
        filter_zone = None
        if feature in ["A norm (pixels*m/s)", "optical flow norm (pixels/s)"]:
            H = Zs.shape[1]
            threshold = H / virtual_height * threshold  # thresholds were computed at H=540
        if isinstance(feature, tuple):
            feature, filter_zone = feature
        if feature in ["angle (deg)"]:
            cvalid = feature_data <= threshold
        else:
            cvalid = feature_data >= threshold

        if filter_zone == 'around_focus_expansion_A':
            A_norm = np.linalg.norm(As, axis=1)
            for ind, mask in enumerate(cvalid):
                num_comp, labels = cv2.connectedComponents((~mask).astype(np.uint8), connectivity=8)
                if num_comp == 0:
                    cvalid[ind].fill(False)
                    continue
                else:
                    focus_expansion_origin = np.unravel_index(np.argmin(A_norm[ind]), A_norm[ind].shape)
                    component_connected_to_origin_label = labels[focus_expansion_origin[0], focus_expansion_origin[1]]
                    mask_component = labels == component_connected_to_origin_label
                    cvalid[ind] = ~np.logical_and(~mask, mask_component)
        valid = np.logical_and(valid, cvalid)

    return valid


def remove_padding(flow, padding):
    left, right, bottom, top = padding
    end_H = flow.shape[1] - bottom
    end_W = flow.shape[2] - right
    return flow[:, top:end_H, left:end_W]


def get_feature_from_depth_from_flow_data(Zs, As, bs, derotating_flows, feature):
    # mask_region = False
    if isinstance(feature, tuple):
        feature, mask_region = feature[:2]
    if feature == 'Z':
        feature_data = Zs
    if feature == "optical flow norm (pixels/s)":
        feature_data = np.linalg.norm(bs, axis=1)
    elif feature == "angle (deg)":
        cos = np.sum(As * bs, axis=1) / np.linalg.norm(As, axis=1) / np.linalg.norm(bs, axis=1)
        cos = np.clip(cos, -1, 1)
        feature_data = np.rad2deg(np.arccos(cos))
    elif feature == "A norm (pixels*m/s)":
        feature_data = np.linalg.norm(As, axis=1)
    # if mask_region == 'top_half':
    #     H, W, = feature_data.shape[-2:]
    #     feature_data[:, int(H // 2):] = np.nan

    return feature_data


def binned_statistic_2D(xs, ys, zs, bins_x, bins_y, callbable, optimize=False, verbose=0):
    # bin i is in interval [bins[i], bins[i+1])
    assigned_bin_x = assign_bins(xs, bins_x)
    assigned_bin_y = assign_bins(ys, bins_y)

    all_stats = {}

    if optimize:
        range_x = np.unique(assigned_bin_x)
        range_x = range_x[np.logical_and(range_x >= 0, range_x <= len(bins_x) - 2)]
    else:
        range_x = range(np.min(assigned_bin_x), len(bins_x) - 1)
    for bin_x in range_x:
        if verbose >= 1:
            print(f'Bin x : {bin_x + 1}/{len(bins_x) - 1}')
        is_in_bin_x = assigned_bin_x == bin_x
        if optimize:
            range_y = np.unique(assigned_bin_y[is_in_bin_x])
            range_y = range_y[np.logical_and(range_y >= 0, range_y <= len(bins_y) - 2)]
        else:
            range_y = range(0, len(bins_y) - 1)
        for bin_y in range_y:
            if verbose >= 2:
                print(f'Bin x : {bin_x + 1}/{len(bins_x) - 1}; Bin y: {bin_y + 1}/{len(bins_y) - 1}')
            mask = np.logical_and(assigned_bin_x == bin_x, assigned_bin_y == bin_y)
            stats = callbable(zs[mask])
            all_stats[bin_x, bin_y] = stats
    return all_stats


def assign_bins(xs, bins_x):
    assigned_bin_x = np.searchsorted(bins_x, xs, side='right') - 1
    assigned_bin_x = np.where(xs == bins_x[-1], assigned_bin_x - 1, assigned_bin_x)
    return assigned_bin_x


def binned_statistic(xs, ys, bins, callable):
    lower = xs < bins[0]
    higher = xs > bins[-1]
    # bin i is in interval [bins[i], bins[i+1])
    assigned_bin = assign_bins(xs, bins)

    all_stats = []
    for bin in range(-1, len(bins)):
        mask = assigned_bin == bin
        if bin == -1:
            assert(np.all(lower == mask))
        elif bin == len(bins) - 1:
            mask = higher
        stats = callable(ys[mask])
        all_stats.append(stats)
    return all_stats


def div_0_fill(a, b, fill_value=np.nan):
    out = np.full_like(a, fill_value, dtype=a.dtype)
    return np.true_divide(a, b, out=out, where=(b != 0))


def save_fig_safe(path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(path)


def safe_name(name):
    return str(name).replace(")", "_").replace("(", "_").replace("*", "x").replace("/", " DIV ")


def truncate_float(f, n):
    '''
    Truncates/pads a float f to n decimal places without rounding
    https://stackoverflow.com/questions/783897/how-to-truncate-float-values
    '''
    if np.isnan(f):
        return "nan"
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def histogram_bounded(name, data, output_path, lims=None, bins=100, figsize=(12, 10)):
    if lims is None:
        lims = (np.min(data), np.max(data))
    below = data < lims[0]
    above = data > lims[1]

    data = data[~np.logical_or(above, below)]

    if isinstance(bins, int) and lims is not None:
        bins = np.linspace(lims[0], lims[1], bins)
    else:
        assert (isinstance(bins, np.ndarray) or isinstance(bins, list))
    h, _ = np.histogram(data, bins)
    h = h / np.sum(h)
    plt.figure(figsize=figsize)
    plt.bar(bins[:-1], h, width=np.diff(bins), align='edge')
    plt.title(f"{name} histogram")

    if output_path is not None:
        save_fig_safe(output_path / safe_name(f"{name} histogram"))

        pd.DataFrame({"min": lims[0], "max": lims[1],
                      "below_count": np.count_nonzero(below),
                      "above_count": np.count_nonzero(above)},
                     index=['stats']).to_csv(output_path / safe_name(f"{name} stats.csv"))


def bin_stats(vals):
    if len(vals) == 0:
        return [np.nan] * 11
    abs_vals = np.abs(vals)
    return np.min(vals), np.max(vals), np.mean(vals), np.std(vals), np.median(vals),\
           np.min(abs_vals), np.max(abs_vals), np.mean(abs_vals), np.std(abs_vals), np.median(abs_vals),\
           len(vals)


stat_to_ind = {
    'min': 0, 'max': 1, 'mean': 2, 'std': 3, 'median': 4,
    'abs_min': 5, 'abs_max': 6, 'abs_mean': 7, 'abs_std': 8, 'abs_median': 9,
    'count': 10
    }

abs_stat_to_ind = {
    'min': 5, 'max': 6, 'mean': 7, 'std': 8, 'median': 9,
    'count': 10
    }
    #               thresholds={
    #                 "Z": 0,
    #                 "angle (deg)": 70, # 85, 78, 70
    # #                   ('Z', None, 'around_focus_expansion_A'): 100,
    #                 "optical flow norm (pixels/s)": 12, # 4, 12
    # #                   "angle (deg)": 80
    #               },
def load_depth(id, id_to_path, id_to_scale):
    data = np.load(id_to_path[id])
    depth = data[data.files[0]]
    scale = id_to_scale.get(id, 1)
    return depth * scale


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


def get_colors(cmap_name='gist_rainbow', number_of_colors=10, xs=None):
    map = cm.get_cmap(cmap_name)
    if xs is None and number_of_colors:
        xs = np.linspace(0, 1, number_of_colors)
    colors = map(MinMaxScaler().fit_transform(xs.reshape(-1, 1)).flatten())
    return colors[:, :3]


LINE_STYLES = ('solid', 'dashed', 'dashdot', 'dotted')
def get_linestyles(number_of_elements, styleset=LINE_STYLES):
    styles = []
    for i in range(number_of_elements):
        styles.append(styleset[i % len(styleset)])
    return styles


def load_first_arr(path):
    raw_data = np.load(path)
    depth = raw_data[raw_data.files[0]]
    return depth


def focal_length_from_flow(centered_points, optical_flow, depths, linear_velocity, angular_velocity,
                           equal_focal_length=True, axes='xy'):
    # assert (not(equal_focal_length and axes != 'xy'))

    b = np.zeros((2 * len(centered_points) + (4 if equal_focal_length else 0)))
    A = np.zeros((2 * len(centered_points) + (4 if equal_focal_length else 0), 6))

    u_dot = optical_flow[:, 0]
    v_dot = optical_flow[:, 1]
    u = centered_points[:, 0]
    v = centered_points[:, 1]
    nu_x, nu_y, nu_z = linear_velocity.T
    omega_x, omega_y, omega_z = angular_velocity.T
    Z = depths

    end = 2 * len(centered_points)
    b[:end:2] = u_dot - u * nu_z / depths
    b[1:end:2] = v_dot - v * nu_z / depths

    A[:end:2, 0] = - nu_x / Z - omega_y
    A[:end:2, 2] = - u ** 2 * omega_y
    A[:end:2, 3] = u * v * omega_x
    A[:end:2, 4] = v * omega_z

    A[1:end:2, 1] = - nu_y / Z + omega_x
    A[1:end:2, 2] = - u * v * omega_y
    A[1:end:2, 3] = v ** 2 * omega_x
    A[1:end:2, 5] = - u * omega_z

    if equal_focal_length:
        b[-1] = 1
        b[-2] = 1
        A[-4, 0] = 1
        A[-4, 1] = -1
        A[-3, 2] = 1
        A[-3, 3] = -1
        A[-2, 4] = 1
        A[-1, 5] = 1

        weight = len(centered_points) / 4 / 100
        b[-4:] = b[-4:] * weight
        A[-4:] = A[-4:] * weight
    if axes != 'xy':
        start = 0 if axes == 'x' else 1
        A = A[start:end:2]
        b = b[start:end:2]
        if equal_focal_length:
            if axes == 'x':
                keep_eq = [-3, -2]
            else:
                keep_eq = [-3, -1]
            A = np.concatenate((A, A[keep_eq, :]), axis=0)
            b = np.concatenate((b, b[keep_eq]), axis=0)

    valid_vars = np.max(np.abs(A[:end]), axis=0) > 1.e-15
    fulL_sol = np.full(6, np.nan)
    if not np.any(valid_vars):
        return fulL_sol
    partial_sol = linear_least_squares(A[:, valid_vars], b)
    fulL_sol[valid_vars] = partial_sol

    return fulL_sol


def focal_length_from_flow_one_f(centered_points, optical_flow, depths, linear_velocity, angular_velocity, axes='xy'):
    A, b, end = prepare_A_b(centered_points, optical_flow, linear_velocity, angular_velocity, depths, axes)

    valid_vars = np.max(np.abs(A[:end]), axis=0) > 1.e-15
    fulL_sol = np.full(2, np.nan)
    if not np.any(valid_vars):
        return fulL_sol
    partial_sol = linear_least_squares(A[:, valid_vars], b)
    fulL_sol[valid_vars] = partial_sol

    map_to_f = np.full_like(fulL_sol, np.nan)
    map_to_f[0] = fulL_sol[0]
    map_to_f[1] = 1/ fulL_sol[1]
    final_f = np.nanmean(map_to_f)
    return fulL_sol, map_to_f, final_f, A, b, valid_vars


def prepare_A_b(centered_points, optical_flow, linear_velocity, angular_velocity, depths, axes):
    b = np.zeros((2 * len(centered_points)))
    A = np.zeros((2 * len(centered_points), 2))
    u_dot = optical_flow[:, 0]
    v_dot = optical_flow[:, 1]
    u = centered_points[:, 0]
    v = centered_points[:, 1]
    nu_x, nu_y, nu_z = linear_velocity.T
    omega_x, omega_y, omega_z = angular_velocity.T
    Z = depths
    end = 2 * len(centered_points)
    b[:end:2] = u_dot - u * nu_z / depths - v * omega_z
    b[1:end:2] = v_dot - v * nu_z / depths + u * omega_z
    A[:end:2, 0] = - nu_x / Z - omega_y
    A[:end:2, 1] = u * v * omega_x - u ** 2 * omega_y
    A[1:end:2, 0] = - nu_y / Z + omega_x
    A[1:end:2, 1] = v ** 2 * omega_x - u * v * omega_y
    if axes != 'xy':
        start = 0 if axes == 'x' else 1
        A = A[start:end:2]
        b = b[start:end:2]
    return A, b, end


def iterative_focal_length_from_flow_one_f(centered_points, optical_flow, depths, linear_velocity, angular_velocity,
                                           initial_value, axes='xy', max_it=100):
    A, b, end = prepare_A_b(centered_points, optical_flow, linear_velocity, angular_velocity, depths, axes)

    previous_sol = initial_value
    for it in range(max_it):
        new_sol = linear_least_squares(A[:, 0:1], b - A[:, 1] * (1 / previous_sol))
        if np.abs(previous_sol - new_sol) < 1.e-10:
            break
        previous_sol = new_sol
    valid_vars = np.ones(2, dtype=bool)
    final_f = previous_sol
    full_sol = np.full(2, final_f)
    full_sol[1] = 1 / final_f
    map_to_f = np.full(2, final_f)
    return full_sol, map_to_f, final_f, A, b, valid_vars
