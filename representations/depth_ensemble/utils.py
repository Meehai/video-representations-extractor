import numpy as np


def scale_depth(reference, target, depth_scaling_clip=(50, 150), depth_scaling_thresh=5, depth_scaling_iter=10):
    mask = np.logical_and(reference > depth_scaling_clip[0], reference < depth_scaling_clip[1])
    if ~np.any(mask):
        return target, -1, mask
    ratio = np.nanmedian(reference[mask]) / np.nanmedian(target[mask])
    target *= ratio

    mask = np.abs(reference - target) < depth_scaling_thresh
    i, last_mask = 0, np.nan
    while np.any(mask != last_mask) and i < depth_scaling_iter and not np.all(~mask):
        r = np.nanmedian(reference[mask]) / np.nanmedian(target[mask])
        target *= r
        ratio *= r
        last_mask = mask
        mask = np.abs(reference - target) < depth_scaling_thresh
        i += 1
        # print(ratio, mask.sum() / np.prod(mask.shape))

    return target, ratio, mask