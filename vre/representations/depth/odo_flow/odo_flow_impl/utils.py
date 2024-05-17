# pylint: disable=all
from pathlib import Path
import numpy as np
import multiprocessing
import torch
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt


def reduce_list(data, mask=None, indices=None):
    assert (mask is None or indices is None)
    if mask is not None:
        return [data[ind] for ind in range(len(mask)) if mask[ind]]
    if indices is not None:
        return [data[ind] for ind in indices]
    return data


def get_params_from_obj(x):
    return dict((key, getattr(x, key)) for key in dir(x) if key not in dir(x.__class__) and not key.startswith("__"))

def is_sorted(ids):
    ids = np.array(ids)
    return np.all((ids[1:] - ids[:-1]) == 1)


def map_timed(func, iterable, size, parallel=True, max_cpu_count=4):
    if max_cpu_count is None:
        max_cpu_count = 0
    if max_cpu_count == 0:
        parallel = False
    if parallel:
        cpu_count = min(max_cpu_count, multiprocessing.cpu_count())
        pool = multiprocessing.Pool(cpu_count)
        cpu_count = multiprocessing.cpu_count()
        chunk_size = max(1, min(100, size // cpu_count))
        for res in tqdm(pool.imap(func, iterable, chunk_size), total=size, smoothing=0):
            yield res
    else:
        for d in tqdm(iterable, total=size, smoothing=0):
            r = func(d)
            yield r


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


class RunningHistogram:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.count_below = 0
        self.count_above = 0
        self.raw_hist = None

    def update(self, data):
        h, self.bins = np.histogram(data, **self.kwargs)
        self.kwargs['bins'] = self.bins
        self.count_below += np.count_nonzero(data < self.bins[0])
        self.count_above += np.count_nonzero(data > self.bins[-1])
        if self.raw_hist is None:
            self.raw_hist = h
        else:
            self.raw_hist += h

    @property
    def hist(self):
        return self.raw_hist / np.sum(self.raw_hist)


class PerFrameStats:
    def __init__(self):
        self.stats = defaultdict(lambda: [])

    def update(self, frame_ind, values):
        for tag, fn in {"mean": np.nanmean, "std": np.nanstd, "median": np.nanmedian,
                        "min": np.nanmin, "max": np.nanmax,
                        "not_nan_pc": lambda x: np.count_nonzero(~np.isnan(x)) / x.size}.items():
            self.stats[tag].append(fn(values))
        self.stats['frame_ind'].append(frame_ind)


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


def load_data_extractor_compatible(path):
    data = np.load(path, allow_pickle=True)
    if not np.issubdtype(data[data.files[0]].dtype, np.number):
        data = data[data.files[0]].item()
        assert 'data' in data
        return data['data']
    return data[data.files[0]].astype(np.float64)


def join_lists(lists):
    result = []
    for l in lists:
        result.extend(l)
    return result


def closest_indexes(source, target):
    source = np.asarray(source)
    target = np.asarray(target)
    closest_point_indexes_right = np.searchsorted(target, source)
    closest_point_indexes_left = closest_point_indexes_right - 1
    closest_point_indexes_left[closest_point_indexes_left < 0] = 0
    closest_point_indexes_right[closest_point_indexes_right >= len(target)] = len(target) - 1
    closest_timestamps_index = np.where(
        np.abs(target[closest_point_indexes_left] - source) <
        np.abs(target[closest_point_indexes_right] - source),
        closest_point_indexes_left, closest_point_indexes_right)
    return closest_timestamps_index


def nn_interpolation(source_t, data, target_t):
    data = data[closest_indexes(source_t, target_t)]
    return data


def flow_warp(flow, target_image):
    H, W = flow.shape[1:]
    displacement = ((np.stack(np.meshgrid(np.arange(W), np.arange(H))) + flow) - np.array(
        ((W - 1) / 2, (H - 1) / 2)).reshape((2, 1, 1))) / np.array(((W - 1) / 2, (H - 1) / 2)).reshape((2, 1, 1))
    displacement = np.clip(displacement, -1, 1)
    displacement_t = torch.unsqueeze(torch.tensor(displacement, dtype=torch.float64).permute(1, 2, 0), dim=0)
    target_t = torch.unsqueeze(torch.tensor(target_image, dtype=torch.float64).permute(2, 0, 1), dim=0)
    frame1_w = torch.squeeze(torch.nn.functional.grid_sample(target_t, displacement_t), dim=0).permute(1, 2, 0).numpy()
    frame1_w = frame1_w.astype(np.uint8)


def auto_depad(data, original_resolution=(3840, 2160)):
    # C X H X W
    H, W = data.shape[-2:]
    W_f = int(round((original_resolution[0] / W)))
    nW = original_resolution[0] // W_f
    H_f = int(round((original_resolution[1] / H)))
    nH = original_resolution[1] // H_f
    assert (nW <= W and nH <= H)
    H_pad = (H - nH) // 2
    W_pad = (W - nW) // 2
    return data[..., H_pad:H_pad + nH, W_pad:W_pad + nW]


def load_first_arr_np_64(path):
    data = np.load(path)
    return data[data.files[0]].astype(np.float64)


def remove_padding(flow, padding):
    left, right, bottom, top = padding
    end_H = flow.shape[1] - bottom
    end_W = flow.shape[2] - right
    return flow[..., top:end_H, left:end_W]


def get_depadder(flow_pad):
    if flow_pad == 'auto':
        flow_depad = auto_depad
    else:
        flow_depad = lambda flow: remove_padding(flow, tuple(map(int, flow_pad)))
    return flow_depad


def get_video_paths(root_dir, extensions, args, pattern="{}{}"):
    extensions = [ext if ext.startswith(".") or len(ext) == 0 else "." + ext for ext in extensions]
    if args.video_file_names is not None:
        all_paths = []
        for name in args.video_file_names:
            for ext in extensions:
                p = Path(root_dir) / pattern.format(name, ext)
                if p.exists() and (ext != '' or p.is_dir()):
                    all_paths.append(p)
        return sorted(all_paths)
    else:
        all_paths = [Path(root_dir).glob(pattern.format(args.name_pattern, ext)) for ext in extensions]
        all_paths = sorted(set(join_lists(all_paths)))
        if args.num_vids is not None:
            all_paths = all_paths[:args.num_vids]
        return all_paths


def show_heatmap(depth):
    plt.figure()
    plt.imshow(depth)
    plt.colorbar()
    plt.show()