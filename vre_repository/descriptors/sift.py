"""sift.py - SIFT in pure numpy.

A drop-in for sift.py with two speed levers:
 1. max_keypoints: descriptors are O(n_keypoints) and ~60% of runtime, so we sort by importance
    and cap to the strongest K BEFORE generateDescriptors. None = all (identical to sift.py); a
    cap returns the exact strongest-K prefix. Pure speed/quality trade, no maths changed.
 2. process_scale: run SIFT on a resized copy (PIL bilinear), then scale keypoint pt/size back to
    the input frame. <1 = fewer pixels = cheaper extrema+gaussian (the per-pixel cost a keypoint
    cap can't touch). SIFT already upsamples 2x internally, so process_scale=0.5 cancels that
    doubling. Coarser keypoint set. process_scale=1.0 = no resize.
"""
# pylint: disable=all
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from vre_video import VREVideo
from vre import Representation, ReprOut, MemoryData
from vre.representations.mixins import NpIORepresentationMixin
from loggez import loggez_logger as logger
from PIL import Image

from .descriptor_representation import DescriptorRepresentation

float_tolerance = 1e-7

@dataclass
class Keypoint:
    """A SIFT keypoint. Drop-in for cv2.KeyPoint; pt is (x, y) in image pixels.
    importance == response (the |DoG| contrast at the extremum): bigger = stronger.
    """
    pt: tuple[float, float] = (0.0, 0.0)
    size: float = 0.0
    angle: float = -1.0
    response: float = 0.0
    octave: int = 0
    class_id: int = -1

    @property
    def importance(self) -> float:
        """keypoint strength used to rank keypoints (higher = keep first for top-k)"""
        return self.response

class SIFT(DescriptorRepresentation):
    """SIFT representation. All computeKeypointsAndDescriptors params live here (with defaults) and
    are passed straight through; set them from vre.yaml `parameters:`."""
    def __init__(self, name: str, dependencies: list[Representation] | None = None,
                 sigma: float = 1.6, num_intervals: int = 3, assumed_blur: float = 0.5,
                 image_border_width: int = 5, max_keypoints: int | None = None,
                 process_scale: float = 1.0):
        DescriptorRepresentation.__init__(self, name=name, dependencies=dependencies)
        self.sigma = sigma
        self.num_intervals = num_intervals
        self.assumed_blur = assumed_blur
        self.image_border_width = image_border_width
        self.max_keypoints = max_keypoints
        self.process_scale = process_scale

    @property
    def n_channels(self):
        """the number of channels"""
        return 128

    def make_images(self, data: ReprOut) -> np.ndarray:
        """image representation of SIFT"""
        breakpoint()
        raise NotImplementedError

    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        """binary representation of SIFT"""
        frames = video[ixs]
        extra = []
        res = np.zeros((len(ixs), ), "object")
        for i, frame in enumerate(frames):
            image_gray = np.array(Image.fromarray(frame).convert("L"))
            kps: list[Keypoint]
            kps, desc = computeKeypointsAndDescriptors(
                image_gray, sigma=self.sigma, num_intervals=self.num_intervals,
                assumed_blur=self.assumed_blur, image_border_width=self.image_border_width,
                max_keypoints=self.max_keypoints, process_scale=self.process_scale)
            res[i] = desc
            extra.append({"frame_size": frame.shape[0:2], "coordinates": [kp.pt for kp in kps],
                          "angles": [kp.angle for kp in kps], "responses": [kp.response for kp in kps],
                          "octave": [kp.octave for kp in kps], "sizes": [kp.size for kp in kps] })
        return ReprOut(frames=frames, output=MemoryData(res), extra=extra, key=ixs)

    def resize(self, data: ReprOut, new_size: tuple[int, int]) -> ReprOut:
        if any(new_size != extra["frame_size"] for extra in data.extra):
            # TODO: resize keypoints to new size (see fastsam).
            raise NotImplementedError(f"{[e['frame_size'] for e in data.extra]} vs {new_size}")
        return data

############################
# numpy cv2 replacements   #
############################

def _gaussian_kernel(sigma: float) -> np.ndarray:
    """1D Gaussian kernel matching cv2.getGaussianKernel(ksize, sigma, CV_32F) with auto ksize.

    cv2 auto ksize for float images is round(sigma*8+1)|1. Values are computed in float64,
    cast to float32, summed in float64, then normalised (matches cv2 bit-for-bit to ~1e-9).
    """
    ksize = int(round(sigma * 8 + 1)) | 1
    c = (ksize - 1) * 0.5
    x = np.arange(ksize, dtype=np.float64) - c
    cf = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
    s = cf.astype(np.float64).sum()
    return (cf.astype(np.float64) / s).astype(np.float32)

def _sep_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable convolution with BORDER_REFLECT_101 (numpy 'reflect'), float32 throughout.

    Each 1D pass is a sliding-window dot with the kernel (BLAS matmul). Matches cv2.GaussianBlur
    to ~1e-4 (only float32 summation order differs), well within the keypoint-detection margin.
    """
    image = image.astype(np.float32, copy=False)
    r = len(kernel) // 2
    p = np.pad(image, ((0, 0), (r, r)), mode="reflect")           # horizontal pass
    out = (sliding_window_view(p, len(kernel), axis=1) @ kernel).astype(np.float32)
    p = np.pad(out, ((r, r), (0, 0)), mode="reflect")             # vertical pass
    return (sliding_window_view(p, len(kernel), axis=0) @ kernel).astype(np.float32)

def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """numpy replacement for cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)"""
    return _sep_convolve(image, _gaussian_kernel(sigma))

def _resize_linear_2x(image: np.ndarray) -> np.ndarray:
    """numpy replacement for cv2.resize(image, (0,0), fx=2, fy=2, INTER_LINEAR). Bit-exact.

    OpenCV maps dst d -> src (d+0.5)/2 - 0.5 (half-pixel centers), clamps to the border.
    For an exact 2x this gives fixed 0.25/0.75 weights, so the float32 result is identical.
    """
    image = image.astype(np.float32, copy=False)
    def _axis(n):
        d = np.arange(2 * n)
        src = (d + 0.5) / 2.0 - 0.5
        i0 = np.floor(src).astype(int)
        frac = (src - i0).astype(np.float32)
        i1 = np.clip(i0 + 1, 0, n - 1)
        return np.clip(i0, 0, n - 1), i1, frac
    i0x, i1x, fx = _axis(image.shape[1])
    tmp = (image[:, i0x] * (1 - fx) + image[:, i1x] * fx).astype(np.float32)
    i0y, i1y, fy = _axis(image.shape[0])
    fy = fy[:, None]
    return (tmp[i0y, :] * (1 - fy) + tmp[i1y, :] * fy).astype(np.float32)

def _resize_nearest_half(image: np.ndarray) -> np.ndarray:
    """numpy replacement for cv2.resize(image, (W//2, H//2), INTER_NEAREST). Bit-exact.

    OpenCV nearest maps dst d -> src floor(d * src/dst), clamped to the last index.
    """
    h, w = image.shape
    dh, dw = int(h / 2), int(w / 2)
    sy = np.minimum(np.floor(np.arange(dh) * (h / dh)).astype(int), h - 1)
    sx = np.minimum(np.floor(np.arange(dw) * (w / dw)).astype(int), w - 1)
    return image[np.ix_(sy, sx)]

#################
# Main function #
#################

def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5,
                                   image_border_width=5, max_keypoints=None, process_scale=1.0):
    """Compute SIFT keypoints and descriptors. Coordinates are always in the INPUT image's frame.

    max_keypoints : cap to the strongest K before descriptors (None = all). process_scale : run on a
    resized copy and map coords back (1.0 = no resize). See the module docstring.
    """
    image = image.astype('float32')
    if process_scale != 1.0:                       # run SIFT on a smaller image, map coords back after
        h, w = image.shape[:2]
        nw, nh = max(1, round(w * process_scale)), max(1, round(h * process_scale))
        image = np.asarray(Image.fromarray(image).resize((nw, nh), Image.BILINEAR), dtype="float32")  # float32 -> mode 'F'
    base_image = generateBaseImage(image, sigma, assumed_blur)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    # Sort by importance (stable; == sort_keypoints_by_importance) then cap, THEN descriptors --
    # descriptors are O(n_keypoints), so capping first is what buys the speedup. Returned set is
    # already importance-sorted, matching sift.py's final sort_keypoints_by_importance.
    keypoints = sorted(keypoints, key=lambda kp: kp.importance, reverse=True)
    if max_keypoints is not None:
        keypoints = keypoints[:max_keypoints]
    descriptors = generateDescriptors(keypoints, gaussian_images)
    if process_scale != 1.0:                       # map pt/size from process scale back to input frame
        inv = 1.0 / process_scale
        for kp in keypoints:
            kp.pt = (kp.pt[0] * inv, kp.pt[1] * inv)
            kp.size *= inv
    return keypoints, descriptors

#########################
# Image pyramid related #
#########################

def generateBaseImage(image, sigma, assumed_blur):
    """Generate base image from input image by upsampling by 2 in both directions and blurring
    """
    logger.debug('Generating base image...')
    image = _resize_linear_2x(image)
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return _gaussian_blur(image, sigma_diff)  # the image blur is now sigma instead of assumed_blur

def computeNumberOfOctaves(image_shape):
    """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
    """
    return int(np.round(np.log(min(image_shape)) / np.log(2) - 1))

def generateGaussianKernels(sigma, num_intervals):
    """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
    """
    logger.debug('Generating scales...')
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = np.zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels

def generateGaussianImages(image, num_octaves, gaussian_kernels):
    """Generate scale-space pyramid of Gaussian images. Returns a list (per octave) of (S, H, W) stacks.
    """
    logger.debug('Generating Gaussian images...')
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = [image]  # first image in octave already has the correct blur
        for gaussian_kernel in gaussian_kernels[1:]:
            image = _gaussian_blur(image, gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(np.stack(gaussian_images_in_octave))
        octave_base = gaussian_images_in_octave[-3]
        image = _resize_nearest_half(octave_base)
    return gaussian_images

def generateDoGImages(gaussian_images):
    """Generate Difference-of-Gaussians image pyramid. Returns a list (per octave) of (S-1, H, W) stacks.
    """
    logger.debug('Generating Difference-of-Gaussian images...')
    return [octave[1:] - octave[:-1] for octave in gaussian_images]

###############################
# Scale-space extrema related #
###############################

def _neighbourhood_max(stack):
    """elementwise 3x3 spatial max per image of a (S, H, W) stack -> (S, H-2, W-2)"""
    v = np.maximum(np.maximum(stack[:, :-2], stack[:, 1:-1]), stack[:, 2:])
    return np.maximum(np.maximum(v[:, :, :-2], v[:, :, 1:-1]), v[:, :, 2:])

def _neighbourhood_min(stack):
    """elementwise 3x3 spatial min per image of a (S, H, W) stack -> (S, H-2, W-2)"""
    v = np.minimum(np.minimum(stack[:, :-2], stack[:, 1:-1]), stack[:, 2:])
    return np.minimum(np.minimum(v[:, :, :-2], v[:, :, 1:-1]), v[:, :, 2:])

def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    """Find pixel positions of all scale-space extrema in the image pyramid (vectorized).

    The 26-neighbour test is a separable 3x3x3 max/min filter (identical candidate set to the
    per-pixel version) and the quadratic-fit localisation is batched over all candidates at once.
    """
    logger.debug('Finding scale-space extrema...')
    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    bw = image_border_width
    keypoints = []

    for octave_index, dog_octave in enumerate(dog_images):
        dog_octave = np.asarray(dog_octave, dtype=np.float32)
        n_dog, h, w = dog_octave.shape
        sp_max = _neighbourhood_max(dog_octave)  # (n_dog, h-2, w-2)
        sp_min = _neighbourhood_min(dog_octave)
        cand_s, cand_i, cand_j = [], [], []
        for image_index in range(1, n_dog - 1):  # middle scales (each has a scale below and above)
            center = dog_octave[image_index, 1:-1, 1:-1]
            nb_max = np.maximum(np.maximum(sp_max[image_index - 1], sp_max[image_index]), sp_max[image_index + 1])
            nb_min = np.minimum(np.minimum(sp_min[image_index - 1], sp_min[image_index]), sp_min[image_index + 1])
            mask = ((center > threshold) & (center == nb_max)) | ((center < -threshold) & (center == nb_min))
            # restrict to [bw, h-bw) x [bw, w-bw); mask index r,c maps to image i,j = r+1,c+1
            mask[:bw - 1] = False; mask[h - bw - 1:] = False
            mask[:, :bw - 1] = False; mask[:, w - bw - 1:] = False
            rc = np.argwhere(mask)
            if rc.size:
                cand_i.append(rc[:, 0] + 1); cand_j.append(rc[:, 1] + 1)
                cand_s.append(np.full(len(rc), image_index))
        if not cand_s:
            continue
        s, i, j = np.concatenate(cand_s), np.concatenate(cand_i), np.concatenate(cand_j)
        pts, sizes, responses, octaves, layers = localizeExtremaBatch(s, i, j, octave_index, num_intervals, dog_octave, sigma, contrast_threshold, image_border_width)
        keypoints.extend(computeKeypointsWithOrientations(pts, sizes, responses, octaves, octave_index, layers, gaussian_images))
    return keypoints

def localizeExtremaBatch(s, i, j, octave_index, num_intervals, dog_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """Batched quadratic-fit refinement of all extrema candidates in one octave.

    Vectorized form of the per-extremum refinement: every candidate iterates together, converged
    ones drop out, out-of-bounds ones are discarded. Uses a pinv solve (the batched equivalent of
    the per-point lstsq; verified to give the identical keypoint set). Convergence on the final
    attempt is rejected, matching the scalar version. Returns the keypoints that pass contrast/edge
    tests as (pts[N,2], sizes[N], responses[N], octaves[N], layers[N]); all in float32 like cv2.
    """
    h, w = dog_octave.shape[1], dog_octave.shape[2]
    s, i, j = s.astype(np.intp), i.astype(np.intp), j.astype(np.intp)
    alive = np.ones(len(s), dtype=bool)
    off = np.array([-1, 0, 1])
    rec_s, rec_i, rec_j, rec_grad, rec_upd, rec_hess, rec_center = [], [], [], [], [], [], []
    for attempt in range(num_attempts_until_convergence):
        a = np.flatnonzero(alive)
        if a.size == 0 or attempt == num_attempts_until_convergence - 1:
            break  # candidates still alive on the final attempt never produce a keypoint
        sa, ia, ja = s[a], i[a], j[a]
        # rescale to [0, 1] (cv2 / Lowe convention); stays float32 under numpy weak promotion
        cube = dog_octave[sa[:, None, None, None] + off[None, :, None, None],
                          ia[:, None, None, None] + off[None, None, :, None],
                          ja[:, None, None, None] + off[None, None, None, :]].astype('float32') / 255.
        gradient = _gradientBatch(cube)
        hessian = _hessianBatch(cube)
        extremum_update = -np.einsum('nij,nj->ni', np.linalg.pinv(hessian), gradient)
        converged = (np.abs(extremum_update) < 0.5).all(axis=1)
        if converged.any():
            c = a[converged]
            rec_s.append(s[c]); rec_i.append(i[c]); rec_j.append(j[c])
            rec_grad.append(gradient[converged]); rec_upd.append(extremum_update[converged])
            rec_hess.append(hessian[converged]); rec_center.append(cube[converged, 1, 1, 1])
            alive[c] = False
        moving = a[~converged]
        if moving.size:
            upd = extremum_update[~converged]
            j[moving] += np.round(upd[:, 0]).astype(np.intp)
            i[moving] += np.round(upd[:, 1]).astype(np.intp)
            s[moving] += np.round(upd[:, 2]).astype(np.intp)
            outside = ((i[moving] < image_border_width) | (i[moving] >= h - image_border_width) |
                       (j[moving] < image_border_width) | (j[moving] >= w - image_border_width) |
                       (s[moving] < 1) | (s[moving] > num_intervals))
            alive[moving[outside]] = False
    if not rec_s:
        return np.empty((0, 2)), np.empty(0), np.empty(0), np.empty(0, np.intp), np.empty(0, np.intp)
    s_, i_, j_ = np.concatenate(rec_s), np.concatenate(rec_i), np.concatenate(rec_j)
    grad, upd, hess, center = np.concatenate(rec_grad), np.concatenate(rec_upd), np.concatenate(rec_hess), np.concatenate(rec_center)
    function_value = center + 0.5 * np.einsum('ni,ni->n', grad, upd)
    contrast_ok = np.abs(function_value) * num_intervals >= contrast_threshold
    trace = hess[:, 0, 0] + hess[:, 1, 1]
    det = np.linalg.det(hess[:, :2, :2])
    edge_ok = (det > 0) & (eigenvalue_ratio * trace ** 2 < (eigenvalue_ratio + 1) ** 2 * det)
    keep = np.flatnonzero(contrast_ok & edge_ok)
    s_, i_, j_, upd, fval = s_[keep], i_[keep], j_[keep], upd[keep], function_value[keep]
    # float32 throughout to match the scalar version (weak int promotion); fed into dedup by ==.
    jf, iff, sf = j_.astype(np.float32), i_.astype(np.float32), s_.astype(np.float32)
    pts = np.stack([(jf + upd[:, 0]) * (2 ** octave_index), (iff + upd[:, 1]) * (2 ** octave_index)], axis=1)
    sizes = sigma * (2 ** ((sf + upd[:, 2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))  # +1 because input was doubled
    octaves = octave_index + s_ * (2 ** 8) + np.round((upd[:, 2] + 0.5) * 255).astype(np.intp) * (2 ** 16)
    return pts, sizes, np.abs(fval), octaves, s_

def _gradientBatch(cube):
    """central-difference gradient at the centre of each (N, 3, 3, 3) cube -> (N, 3)"""
    dx = 0.5 * (cube[:, 1, 1, 2] - cube[:, 1, 1, 0])
    dy = 0.5 * (cube[:, 1, 2, 1] - cube[:, 1, 0, 1])
    ds = 0.5 * (cube[:, 2, 1, 1] - cube[:, 0, 1, 1])
    return np.stack([dx, dy, ds], axis=1)

def _hessianBatch(cube):
    """central-difference Hessian at the centre of each (N, 3, 3, 3) cube -> (N, 3, 3)"""
    c = cube[:, 1, 1, 1]
    dxx = cube[:, 1, 1, 2] - 2 * c + cube[:, 1, 1, 0]
    dyy = cube[:, 1, 2, 1] - 2 * c + cube[:, 1, 0, 1]
    dss = cube[:, 2, 1, 1] - 2 * c + cube[:, 0, 1, 1]
    dxy = 0.25 * (cube[:, 1, 2, 2] - cube[:, 1, 2, 0] - cube[:, 1, 0, 2] + cube[:, 1, 0, 0])
    dxs = 0.25 * (cube[:, 2, 1, 2] - cube[:, 2, 1, 0] - cube[:, 0, 1, 2] + cube[:, 0, 1, 0])
    dys = 0.25 * (cube[:, 2, 2, 1] - cube[:, 2, 0, 1] - cube[:, 0, 2, 1] + cube[:, 0, 0, 1])
    return np.stack([np.stack([dxx, dxy, dxs], axis=1),
                     np.stack([dxy, dyy, dys], axis=1),
                     np.stack([dxs, dys, dss], axis=1)], axis=1)

#########################
# Keypoint orientations #
#########################

def computeKeypointsWithOrientations(pts, sizes, responses, octaves, octave_index, layers, gaussian_images,
                                     radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Assign orientations to a batch of keypoints from one octave (one keypoint per dominant peak).

    The gradient-orientation histograms are built for all keypoints at once, grouped by (layer,
    radius) so each group shares an image and a sample grid. Accumulation stays in keypoint-major
    scan order (np.nonzero on the (n_kp, n_sample) mask), so each histogram is bit-identical to the
    per-keypoint version. Peak detection / quadratic interpolation is then done per keypoint.
    """
    n = len(layers)
    if n == 0:
        return []
    scale = scale_factor * sizes / np.float32(2 ** (octave_index + 1))  # compare with kp_size in localizeExtremaBatch()
    radius = np.round(radius_factor * scale).astype(int)
    weight_factor = -0.5 / (scale ** 2)
    base_y = np.round(pts[:, 1] / np.float32(2 ** octave_index)).astype(int)
    base_x = np.round(pts[:, 0] / np.float32(2 ** octave_index)).astype(int)
    raw = np.zeros((n, num_bins))

    groups = {}
    for idx in range(n):
        groups.setdefault((int(layers[idx]), int(radius[idx])), []).append(idx)
    for (layer, rad), members in groups.items():
        gaussian_image = gaussian_images[octave_index][layer]
        h, w = gaussian_image.shape
        mem = np.array(members)
        offs = np.arange(-rad, rad + 1)              # shared (i, j) grid in scan order (i outer, j inner)
        ii, jj = np.meshgrid(offs, offs, indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        ry = base_y[mem][:, None] + ii[None, :]      # (nb, ns)
        rx = base_x[mem][:, None] + jj[None, :]
        valid = (ry > 0) & (ry < h - 1) & (rx > 0) & (rx < w - 1)
        kk, ss = np.nonzero(valid)                   # keypoint-major, scan order within each keypoint
        if len(kk) == 0:
            continue
        ryv, rxv = ry[kk, ss], rx[kk, ss]
        dx = gaussian_image[ryv, rxv + 1] - gaussian_image[ryv, rxv - 1]
        dy = gaussian_image[ryv - 1, rxv] - gaussian_image[ryv + 1, rxv]
        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
        weight = np.exp(weight_factor[mem][kk] * (ii[ss] ** 2 + jj[ss] ** 2))
        histogram_index = np.round(gradient_orientation * num_bins / 360.).astype(int) % num_bins
        np.add.at(raw, (mem[kk], histogram_index), weight * gradient_magnitude)

    # circular [1, 4, 6, 4, 1] / 16 smoothing per keypoint (np.roll(h, k)[n] == h[n - k]); bit-identical to the loop
    smooth = (6 * raw + 4 * (np.roll(raw, 1, axis=1) + np.roll(raw, -1, axis=1))
              + np.roll(raw, 2, axis=1) + np.roll(raw, -2, axis=1)) / 16.
    orientation_max = smooth.max(axis=1)
    is_peak = (smooth > np.roll(smooth, 1, axis=1)) & (smooth > np.roll(smooth, -1, axis=1))
    keypoints = []
    for k in range(n):
        sh = smooth[k]
        for peak_index in np.flatnonzero(is_peak[k]):
            peak_value = sh[peak_index]
            if peak_value >= peak_ratio * orientation_max[k]:
                # Quadratic peak interpolation, eqn (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
                left_value = sh[(peak_index - 1) % num_bins]
                right_value = sh[(peak_index + 1) % num_bins]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360. - interpolated_peak_index * 360. / num_bins
                if abs(orientation - 360.) < float_tolerance:
                    orientation = 0
                keypoints.append(Keypoint(pt=(pts[k, 0], pts[k, 1]), size=sizes[k], angle=orientation,
                                          response=responses[k], octave=int(octaves[k])))
    return keypoints

##############################
# Duplicate keypoint removal #
##############################

def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

#############################
# Keypoint scale conversion #
#############################

def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

#########################
# Descriptor generation #
#########################

def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint.

    Keypoints are grouped into buckets sharing (octave, layer, half_width) -- each bucket has the
    same source image and sample grid, so all its keypoints are processed in one vectorized pass.
    The 8-corner trilinear scatter is eight np.bincount calls (no np.stack, no np.add.at); bincount
    is a tight C loop and the summation order differs from the per-sample loop only in float
    rounding -- verified bit-identical to sift.py's descriptors on the reference frame.
    """
    logger.debug('Generating descriptors...')
    n = len(keypoints)
    out = np.zeros((n, window_width * window_width * num_bins), dtype="float32")
    if n == 0:
        return out
    bins_per_degree = num_bins / 360.
    weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
    ncols = window_width + 2

    # group keypoints by (octave, layer, half_width); a group shares image + grid shape
    groups = {}
    for idx, kp in enumerate(keypoints):
        octave, layer, scale = unpackOctave(kp)
        num_rows, num_cols = gaussian_images[octave + 1][layer].shape
        hist_width = scale_multiplier * 0.5 * scale * kp.size
        half_width = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))  # diagonal of the window
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))      # keep within image
        groups.setdefault((octave, layer, half_width), []).append((idx, kp, scale, hist_width))

    for (octave, layer, half_width), members in groups.items():
        gaussian_image = gaussian_images[octave + 1][layer]
        num_rows, num_cols = gaussian_image.shape
        idxs = np.array([m[0] for m in members])
        angle = np.array([360. - m[1].angle for m in members])                  # (nb,)
        cos_angle, sin_angle = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        hist_width = np.array([m[3] for m in members])                          # (nb,)
        point = np.array([np.round(m[2] * np.array(m[1].pt)).astype(int) for m in members])  # (nb, 2) = (x, y)

        offs = np.arange(-half_width, half_width + 1)                           # shared grid, scan order
        rows, cols = np.meshgrid(offs, offs, indexing="ij")
        rows, cols = rows.ravel().astype(np.float32), cols.ravel().astype(np.float32)
        row_rot = cols[None, :] * sin_angle[:, None] + rows[None, :] * cos_angle[:, None]   # (nb, ns)
        col_rot = cols[None, :] * cos_angle[:, None] - rows[None, :] * sin_angle[:, None]
        row_bin = row_rot / hist_width[:, None] + 0.5 * window_width - 0.5
        col_bin = col_rot / hist_width[:, None] + 0.5 * window_width - 0.5
        window_row = np.round(point[:, 1][:, None] + rows[None, :]).astype(int)
        window_col = np.round(point[:, 0][:, None] + cols[None, :]).astype(int)
        valid = (row_bin > -1) & (row_bin < window_width) & (col_bin > -1) & (col_bin < window_width) & \
                (window_row > 0) & (window_row < num_rows - 1) & (window_col > 0) & (window_col < num_cols - 1)
        kk, ss = np.nonzero(valid)   # keypoint-major, scan order within each keypoint
        if len(kk) == 0:
            continue
        row_bin, col_bin = row_bin[kk, ss], col_bin[kk, ss]
        row_rot, col_rot = row_rot[kk, ss], col_rot[kk, ss]
        hw = hist_width[kk]
        wr, wc = window_row[kk, ss], window_col[kk, ss]
        dx = gaussian_image[wr, wc + 1] - gaussian_image[wr, wc - 1]
        dy = gaussian_image[wr - 1, wc] - gaussian_image[wr + 1, wc]
        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
        weight = np.exp(weight_multiplier * ((row_rot / hw) ** 2 + (col_rot / hw) ** 2))
        magnitude = weight * gradient_magnitude
        orientation_bin = (gradient_orientation - angle[kk]) * bins_per_degree

        # trilinear interpolation (inverse): distribute each sample over its 8 histogram corners
        row_bin_floor = np.floor(row_bin).astype(int)
        col_bin_floor = np.floor(col_bin).astype(int)
        orientation_bin_floor = np.floor(orientation_bin).astype(int)
        row_fraction = row_bin - row_bin_floor
        col_fraction = col_bin - col_bin_floor
        orientation_fraction = orientation_bin - orientation_bin_floor
        orientation_bin_floor = np.where(orientation_bin_floor < 0, orientation_bin_floor + num_bins, orientation_bin_floor)
        orientation_bin_floor = np.where(orientation_bin_floor >= num_bins, orientation_bin_floor - num_bins, orientation_bin_floor)

        c1 = magnitude * row_fraction
        c0 = magnitude * (1 - row_fraction)
        c11 = c1 * col_fraction
        c10 = c1 * (1 - col_fraction)
        c01 = c0 * col_fraction
        c00 = c0 * (1 - col_fraction)
        c111 = c11 * orientation_fraction
        c110 = c11 * (1 - orientation_fraction)
        c101 = c10 * orientation_fraction
        c100 = c10 * (1 - orientation_fraction)
        c011 = c01 * orientation_fraction
        c010 = c01 * (1 - orientation_fraction)
        c001 = c00 * orientation_fraction
        c000 = c00 * (1 - orientation_fraction)

        # 8 corners per sample scattered with bincount; rc?? are the 4 (row,col) bases (vs 8 flat() calls)
        rbf, cbf, obf = row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor
        ob1 = (obf + 1) % num_bins
        base = kk * (ncols * ncols * num_bins)   # per-keypoint slot in the flat histogram
        rc00 = (rbf * ncols + cbf) * num_bins + base
        rc01 = (rbf * ncols + cbf + 1) * num_bins + base
        rc10 = ((rbf + 1) * ncols + cbf) * num_bins + base
        rc11 = ((rbf + 1) * ncols + cbf + 1) * num_bins + base
        L = len(members) * ncols * ncols * num_bins
        histogram = (np.bincount(rc00 + obf, c000, L) + np.bincount(rc00 + ob1, c001, L) +
                     np.bincount(rc01 + obf, c010, L) + np.bincount(rc01 + ob1, c011, L) +
                     np.bincount(rc10 + obf, c100, L) + np.bincount(rc10 + ob1, c101, L) +
                     np.bincount(rc11 + obf, c110, L) + np.bincount(rc11 + ob1, c111, L))
        histogram = histogram.reshape(len(members), ncols, ncols, num_bins)

        descriptor = histogram[:, 1:-1, 1:-1, :].reshape(len(members), -1)  # remove histogram borders
        # Threshold and normalize per keypoint
        threshold = np.linalg.norm(descriptor, axis=1, keepdims=True) * descriptor_max_value
        descriptor = np.minimum(descriptor, threshold)
        descriptor = descriptor / np.maximum(np.linalg.norm(descriptor, axis=1, keepdims=True), float_tolerance)
        # Multiply by 512, round, and saturate between 0 and 255 (OpenCV convention)
        out[idxs] = np.clip(np.round(512 * descriptor), 0, 255).astype("float32")
    return out

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--max_keypoints", type=int)
    parser.add_argument("--process_scale", type=float, default=1)
    args = parser.parse_args()
    if args.image_path.suffix == ".npz":
        image = np.load(args.image_path)["arr_0"]
    else:
        image = np.array(Image.open(args.image_path))

    sift = SIFT("sift", max_keypoints=args.max_keypoints, process_scale=args.process_scale)
    res = sift.compute(image[None], ixs=[0])
    extra, desc = res.extra[0], res.output[0]
    pts = np.array(extra["coordinates"])

    # sanity check with a fixed image in test/e2e/data/rgb/0.npz -- defaults reproduce sift.py
    if np.allclose(image.mean(), 69.36927854938271) and np.allclose(image.std(), 36.73498952105777):
        desc: np.ndarray
        assert len(pts) == 3226, len(pts)
        assert np.allclose(desc.mean(), 27.259924), desc.mean()
        assert np.allclose(desc.std(), 36.124), desc.mean()
        assert np.allclose(pts.mean(0), [407.26937201, 244.16914445]), pts.mean(0)
        assert np.allclose(pts.std(0), [239.59762926, 131.16306772]), pts.std(0)
