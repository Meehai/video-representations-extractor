# pylint: disable=all
import numpy as np
import torch
from scipy import ndimage as ndi

from .polyfit import linear_least_squares


def solve_delta_ang_vel_old(jacobian_z, jacobian_ang_vel, b):
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


def solve_delta_ang_vel(jacobian_z, jacobian_ang_vel, b):
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


def depth_from_flow_triangulate(source, dst, P1, P2):
    A = np.zeros((len(source), 4, 4))
    A[:, 0] = source[:, 0][:, None] * P1[2, :] - P1[0, :]
    A[:, 1] = source[:, 1][:, None] * P1[2, :] - P1[1, :]
    A[:, 2] = dst[:, 0][:, None] * P2[2, :] - P2[0, :]
    A[:, 3] = dst[:, 1][:, None] * P2[2, :] - P2[1, :]
    A = A / np.linalg.norm(A, axis=-1)[:, :, None]
    u, s, vh = np.linalg.svd(A, )
    sol = vh[:, -1]
    pts = sol[:, :3] / sol[:, 3:]
    Z = pts[:, 2]
    return Z


def depth_from_flow_triangulate_multi_dst(source, dsts, P1, P2s, solutions=["overall", "frame", "both"][1],
                                          third_eq=True):
    n = 3 if third_eq else 2
    A = np.zeros((len(source), n * (len(dsts) + 1), 4))
    A[:, 0] = source[:, 0][:, None] * P1[2, :] - P1[0, :]
    A[:, 1] = source[:, 1][:, None] * P1[2, :] - P1[1, :]
    if n == 3:
        A[:, 2] = source[:, 0][:, None] * P1[1, :] - source[:, 1][:, None] * P1[0, :]
    for i in range(len(dsts)):
        A[:, n + n * i + 0] = dsts[i, :, 0][:, None] * P2s[i][2, :] - P2s[i][0, :]
        A[:, n + n * i + 1] = dsts[i, :, 1][:, None] * P2s[i][2, :] - P2s[i][1, :]
        if third_eq:
            A[:, n + n * i + 2] = dsts[i, :, 0][:, None] * P2s[i][1, :] - dsts[i, :, 1][:, None] * P2s[i][0, :]
    A_norm = np.linalg.norm(A, axis=-1)[:, :, None]

    A = np.divide(A, A_norm, out=A, where=A_norm > 0)

    Z = None
    Zs = None
    if solutions in ["overall", "both"]:
        Z = triangulate_point_from_homogeneous_system(A)
    if solutions in ["frame", "both"]:
        Zs = []
        if len(dsts) > 1 or Z is None:
            for i in range(len(dsts)):
                if third_eq:
                    Zi = triangulate_point_from_homogeneous_system(
                        A[:, (0, 1, 2, 3 + 3 * i, 3 + 3 * i + 1, 3 + 3 * i + 2)])
                else:
                    Zi = triangulate_point_from_homogeneous_system(A[:, (0, 1, 2 + 2 * i, 2 + 2 * i + 1)])
                Zs.append(Zi)
        else:
            Zs = [Z]
    return Z, Zs


# print('Jax warm-up: ', timeit.timeit(lambda: jax.numpy.linalg.svd(np.random.random((10, 10, 3, 3)).astype(dtype=np.float64)), number=1))

def triangulate_point_from_homogeneous_system(A):
    # EIG
    # e, ev = np.linalg.eigh(np.transpose(A, (0, 2, 1)) @ A)
    # sol = np.squeeze(np.take_along_axis(ev, np.expand_dims(np.argmin(e, axis=1), (1, 2)), axis=2), axis=2)

    # e, ev = jax.numpy.linalg.eigh(np.transpose(A, (0, 2, 1)) @ A)
    # sol = jax.numpy.squeeze(jax.numpy.take_along_axis(ev, jax.numpy.expand_dims(jax.numpy.argmin(e, axis=1), (1, 2)), axis=2), axis=2)
    # sol = np.array(sol)
    #
    # u, s, vh = jax.numpy.linalg.svd(A)
    # vh = np.array(vh)
    # sol = vh[:, -1]
    #
    u, s, vh = np.linalg.svd(A, )
    sol = vh[:, -1]

    # with torch.no_grad():
    #     u, s, vh = torch.linalg.svd(torch.Tensor(A).to(torch.device("cpu")))
    #     vh = vh.cpu().numpy()
    #     sol = vh[:, -1]

    # A = torch.Tensor(A).to(torch.device("cpu"))
    # e, ev = torch.linalg.eigh(torch.transpose(A, 2, 1) @ A)
    # e = e.cpu().numpy()
    # ev = ev.cpu().numpy()
    # sol = np.squeeze(np.take_along_axis(ev, np.expand_dims(np.argmin(e, axis=1), (1, 2)), axis=2), axis=2)

    pts = sol[:, :3] / sol[:, 3:]
    Z = pts[:, 2]
    return Z


def depth_from_flow_all(source, dsts, K, pose2=None, velocities=None, dts=None,
                        solutions=["overall", "frame", "both"][1], pose_type=["pose", "velocities", "both"][0],
                        method=["Z", "P_SVD"][0], third_eq=False):
    if method == "P_SVD":
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2s = [K @ np.hstack((R, t[:, None])) for R, t in pose2]
        Z, Zs = depth_from_flow_triangulate_multi_dst(source, dsts, P1, P2s, solutions, third_eq)
        return Z, Zs, None
    elif method == "Z":
        assert not third_eq, "third_eq option only for P_SVD method"
        return depth_from_flow_both(source, dsts, K, pose2, velocities, dts, solutions, pose_type)


def depth_from_flow_both(source, dsts, K, pose2=None, velocities=None, dts=None, solutions=["overall", "frame", "both"],
                         type=["pose", "velocities", "both"][2]):
    N_points = len(source)
    N_frames = len(pose2)

    fx, fy, u0, v0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fs = np.array((fx, fy))
    center = np.array((u0, v0))

    As = []
    bs = []

    if type in ["pose", "both"]:
        assert pose2 is not None, "Target poses are required"
        p0 = (source - center) / fs
        ps = (dsts - center) / fs
        ps_h = np.concatenate((ps, np.ones((*ps.shape[:2], 1))), axis=-1)
        p0_h = np.concatenate((p0, np.ones((*p0.shape[:1], 1))), axis=-1)

        Rs, ts = zip(*pose2)
        ts = np.stack(ts, axis=0)
        Rs = np.stack(Rs, axis=0)

        # N_frames x N_points x 3
        bp = - np.cross(ps_h, np.expand_dims(ts, axis=1))

        # N_points x 3
        Q = np.transpose(Rs @ p0_h.T, (0, 2, 1))

        # N_frames x N_points x 3
        Ap = np.cross(ps_h, Q)
        As.append(Ap)
        bs.append(bp)

    if type in ["velocities", "both"]:
        assert velocities is not None and dts is not None, "Velocities and time-deltas are required"
        # add inst equatons based on jacobian
        lin_vel, ang_vel = velocities
        uv_bar = source - center
        J_w = get_J_w(uv_bar[:, 0], uv_bar[:, 1], fx, fy)
        derotating_flow = np.transpose(J_w, (2, 0, 1)).dot(ang_vel)
        Av = uv_bar * lin_vel[2] - fs * lin_vel[:2]
        Avs, bvs = [], []
        for dst, dt in zip(dsts, dts):
            bv = (dst - source) / dt - derotating_flow
            Avs.append(Av)
            bvs.append(bv)
        Avs = np.stack(Avs)
        bvs = np.stack(bvs)
        # original system solves for 1/Z
        As.append(bvs)
        bs.append(Avs)

    A = np.concatenate(As, axis=-1)
    b = np.concatenate(bs, axis=-1)

    # if masks is not None:
    #     A[~masks] = 0
    #     b[~masks] = 0
    # if weights is not None:
    #     weights = np.expand_dims(weights, axis=-1)
    #     A = A * weights
    #     b = b * weights

    Z = None
    Zs = None
    if solutions in ["overall", "both"]:
        A_all = np.transpose(A, (1, 0, 2)).reshape(N_points, -1)
        b_all = np.transpose(b, (1, 0, 2)).reshape(N_points, -1)
        Z = solve_scalar_linear_least_squares(A_all, b_all)
    if solutions in ["frame", "both"]:
        if len(dsts) > 1 or Z is None:
            Zs = solve_scalar_linear_least_squares(A, b)
            # Z, certainty = aggregate_depths(A, Z, Zs, aggregation, b, weights)
        else:
            Zs = Z
    return Z, Zs, None


def aggregate_depths(A, Z, Zs, aggregation, b, weights):
    unit_vectors = False
    if aggregation == "mean":
        Z = np.nanmean(Zs, axis=0)
        certainty = None
    elif aggregation == "median":
        Z = np.nanmedian(Zs, axis=0)
        certainty = None
    elif aggregation == "weighted_mean_sine_sq":
        assert unit_vectors
        sine_sq = np.sum(A ** 2, axis=-1)
        weights = sine_sq
        Z = weighted_Z(Zs, weights)
        certainty = np.rad2deg(np.arcsin(np.sqrt(np.mean(sine_sq, axis=0))))
    elif aggregation == "weighted_mean_sine":
        assert unit_vectors
        sin = np.linalg.norm(A, axis=-1)
        weights = sin
        Z = weighted_Z(Zs, weights)
        certainty = np.rad2deg(np.arcsin(np.mean(sin, axis=0)))
    elif aggregation == "weighted_mean_sine_over_err":
        assert unit_vectors
        sin_sq = np.sum(A ** 2, axis=-1)
        err_sq = np.sum((A * Zs[:, :, None] - b) ** 2, axis=-1)
        weights = np.sqrt(sin_sq / err_sq)
        Z = weighted_Z(Zs, weights)
        certainty = np.rad2deg(np.arcsin(np.sqrt(np.mean(sin_sq, axis=0))))
    elif aggregation == "weighted_mean_sine_sq_over_err_sq":
        assert unit_vectors
        sin_sq = np.sum(A ** 2, axis=-1)
        err_sq = np.sum((A * Zs[:, :, None] - b) ** 2, axis=-1)
        weights = sin_sq / err_sq
        Z = weighted_Z(Zs, weights)
        certainty = np.rad2deg(np.arcsin(np.sqrt(np.mean(sin_sq, axis=0))))
    elif aggregation == "weighted_mean_sine_sq_over_err":
        assert unit_vectors
        sin_sq = np.sum(A ** 2, axis=-1)
        err = np.linalg.norm((A * Zs[:, :, None] - b) ** 2, axis=-1)
        weights = sin_sq / err
        Z = weighted_Z(Zs, weights)
        certainty = np.rad2deg(np.arcsin(np.sqrt(np.mean(sin_sq, axis=0))))
    elif aggregation == "weighted_mean_sine_sq_weights":
        assert unit_vectors
        sin_sq = np.sum(A ** 2, axis=-1)
        weights = sin_sq * np.minimum(1, weights)
        Z = weighted_Z(Zs, weights)
        certainty = np.rad2deg(np.arcsin(np.sqrt(np.mean(sin_sq, axis=0))))
    else:
        assert False, f"Unknown depth aggregation method: {aggregation}"
    return Z, certainty


def weighted_Z(Zs, weights):
    invalid = np.isnan(Zs)
    weights[invalid] = 0
    Z = np.nansum(Zs * weights, axis=0) / np.sum(weights, axis=0)
    return Z


def solve_scalar_linear_least_squares(A, b):
    A_norm_sq_all = np.sum(np.square(A), axis=-1)
    Ab_all = np.sum(A * b, axis=-1)
    # N_frames x N_points
    Z = Ab_all / A_norm_sq_all
    return Z


def depth_from_flow(batched_flow, linear_velocity, angular_velocity, K, adjust_ang_vel=True, use_focus_correction=False,
                    use_cosine_correction_gd=True, mesh_grid=None, axis=("x", "y", "xy")[2]):
    f_u, f_v = K[0, 0], K[1, 1]
    u0, v0 = K[0, 2], K[1, 2]

    H, W = batched_flow.shape[-2:]

    if mesh_grid is not None:
        us_bar, vs_bar = mesh_grid
    else:
        us_bar, vs_bar = init_mesh_grid(H, W, u0, v0)

    B = len(linear_velocity)

    # compute J_w
    # J_w = np.array([
    #     [u * v / f_v, -(f_u * f_u + u * u) / f_u, f_u / f_v * v],
    #     [(f_v * f_v + v * v) / f_v, -u * v / f_u, - f_v / f_u * u],
    # ])

    # compute A = J_t @ sensor_velocity
    # J_t = np.array([
    #     [-f_u, 0, u],
    #     [0, -f_v, v]])

    A = get_A(linear_velocity, us_bar, vs_bar, f_u, f_v)

    # derotating_flow = np.empty((B, 2, H, W))
    angular_velocity = angular_velocity.copy()

    J_w = get_J_w(us_bar, vs_bar, f_u, f_v)

    derotating_flow = get_derotating_flow(J_w, angular_velocity)

    # compute b
    b = batched_flow - derotating_flow
    if adjust_ang_vel:
        step = int(80 * H / 540)  # for 12x7 grid at 540x960

        for b_ind in range(B):
            mask = np.ones_like(us_bar, dtype=bool)
            mask_sample = mask[::step, ::step]
            # mask_sample.fill(True)
            A_sample = A[b_ind][:, ::step, ::step][:, mask_sample]
            b_sample = b[b_ind][:, ::step, ::step][:, mask_sample]
            J_w_sample = J_w[:, :, ::step, ::step][:, :, mask_sample]
            solution = solve_delta_ang_vel(A_sample, J_w_sample, b_sample)
            if solution is not None:
                ang_vel_correction = solution[-3:]
                angular_velocity[b_ind] = angular_velocity[b_ind] + ang_vel_correction
        derotating_flow = get_derotating_flow(J_w, angular_velocity)
        b = batched_flow - derotating_flow

    ang_vel_correction = np.zeros_like(angular_velocity)
    if use_focus_correction:
        for b_ind in range(B):
            ang_vel_correction[b_ind] = focus_corection(angular_velocity[b_ind], linear_velocity[b_ind],
                                                        f_u, (int(u0), int(v0)), b[b_ind])
    if use_cosine_correction_gd:
        for b_ind in range(B):
            ang_vel_correction[b_ind] = cosine_correction_torch(b[b_ind], A[b_ind], J_w,
                                                                initial_delta=ang_vel_correction[b_ind], b_ind=b_ind)

    angular_velocity = angular_velocity - ang_vel_correction
    derotating_flow = get_derotating_flow(J_w, angular_velocity)
    b = batched_flow - derotating_flow

    if axis == "xy":
        norm_A_squared = np.sum(np.square(A), axis=1)
        dot_Ab = np.sum(A * b, axis=1)
        Z = np.divide(norm_A_squared, dot_Ab, out=np.full_like(dot_Ab, np.nan, dtype=dot_Ab.dtype), where=dot_Ab != 0)
    elif axis == "x":
        Z = A[:, 0] / b[:, 0]
    else:
        Z = A[:, 1] / b[:, 1]
    return Z, A, b, derotating_flow, angular_velocity


def get_J_w(us_bar, vs_bar, f_u, f_v):
    us_bar_sq = us_bar ** 2
    vs_bar_sq = vs_bar ** 2
    uv = us_bar * vs_bar
    J_w = np.array([[uv / f_v, - (f_u ** 2 + us_bar_sq) / f_u, f_u / f_v * vs_bar],
                    [(f_v ** 2 + vs_bar_sq) / f_v, - uv / f_u, - f_v / f_u * us_bar]])
    return J_w


def compute_zero_ang_vel_delta(ang_vel, lin_vel, f, c, b):
    # speed check
    low_axis_threshold = 0.1
    low_ang_vel_threshold = 0.05
    k_max = 0
    use_focus_correction = True

    if use_focus_correction:
        focus_correction_valid = True

        # check if we can approximante ang vel along one axis to zero
        # ang_low_idx = np.argmin(np.abs(ang_vel))
        ang_low_idx = 1
        if np.abs(ang_vel).mean() > low_ang_vel_threshold and np.abs(ang_vel).min() / np.abs(
                ang_vel).max() > low_axis_threshold:
            focus_correction_valid = False

        # check if focus of expansion is inside the image space
        u, v = np.round(f * lin_vel[0:2] / lin_vel[2])
        if not np.isfinite(u) or not np.isfinite(v):
            focus_correction_valid = False
        elif u < -c[0] or u > c[0] - 1 or v < -c[1] or v > c[1] - 1:
            focus_correction_valid = False

        # TODO check real min vs virtual min

        # if looking good
        if focus_correction_valid:
            ku = np.max([k_max - np.abs(u), 0])
            kv = np.max([k_max - np.abs(v), 0])

            x = np.array([np.max([u - ku, -c[0]]), np.max([u - ku, -c[0]]),
                          np.min([u + ku, c[0]]), np.min([u + ku, c[0]])])
            y = np.array([np.max([v - kv, -c[1]]), np.min([v + kv, c[1]]),
                          np.max([v - kv, -c[1]]), np.min([v + kv, c[1]])])

            xy = np.stack((x, y), axis=1)
            xy = np.unique(xy, axis=0).astype(np.int32)

            x, y = x.astype(np.int), y.astype(np.int)
            jw = np.array([np.array([[u * v / f, -(f ** 2 + u ** 2) / f, v],
                                     [(f ** 2 + v ** 2) / f, -u * v / f, -u]]) for u, v in xy])

            jw = np.delete(jw, ang_low_idx, axis=2)
            jw = jw.reshape(jw.shape[0] * 2, 2)

            tg = np.array([-b[:, v + c[1], u + c[0]] for u, v in xy])
            tg = tg.flatten()

            try:
                dw = np.linalg.lstsq(jw, tg, rcond=None)[0]
                dw = np.insert(dw, ang_low_idx, 0, axis=0)
            except np.linalg.LinAlgError:
                dw = np.zeros(3)
        else:
            dw = np.zeros(3)
    else:
        dw = np.zeros(3)

    return dw


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
    vs, us = np.meshgrid(vs, us, indexing="ij")
    us_bar = us - u0
    vs_bar = vs - v0
    return us_bar, vs_bar


def get_derotating_flow(J_w, angular_velocity):
    angular_velocity = np.expand_dims(angular_velocity, axis=(2, 3))
    derotating_flow = np.transpose((
            np.transpose(J_w, (2, 3, 0, 1)) @ np.transpose(angular_velocity, (2, 3, 1, 0))),
        (3, 2, 0, 1))
    return derotating_flow


def filter_depth_from_flow(Zs, As, bs, derotating_flows, thresholds, virtual_height=540):
    import cv2
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

        if filter_zone == "around_focus_expansion_A":
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


def get_feature_from_depth_from_flow_data(Zs, As, bs, derotating_flows, feature):
    # mask_region = False
    if isinstance(feature, tuple):
        feature, mask_region = feature[:2]
    if feature == "Z":
        feature_data = Zs
    if feature == "optical flow norm (pixels/s)":
        feature_data = np.linalg.norm(bs, axis=1)
    elif feature == "angle (deg)":
        with np.errstate(divide="ignore"):
            As_norm = np.linalg.norm(As, axis=1)
            bs_norm = np.linalg.norm(bs, axis=1)
            dot_Ab = np.sum(As * bs, axis=1)
            Ab_norm = As_norm * bs_norm
            cos = np.divide(dot_Ab, Ab_norm, out=np.full_like(dot_Ab, np.nan, dtype=dot_Ab.dtype), where=Ab_norm > 0)
        cos = np.clip(cos, -1, 1)
        feature_data = np.rad2deg(np.arccos(cos))
    elif feature == "A norm (pixels*m/s)":
        feature_data = np.linalg.norm(As, axis=1)
    # if mask_region == "top_half":
    #     H, W, = feature_data.shape[-2:]
    #     feature_data[:, int(H // 2):] = np.nan

    return feature_data

def gaussian(
    image,
    sigma=1,
    mode='nearest',
    cval=0,
    preserve_range=False,
    truncate=4.0,
    *,
    channel_axis=None,
    out=None,
):
    """Multi-dimensional Gaussian filter."""
    if np.any(np.asarray(sigma) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")
    if channel_axis is not None:
        # do not filter across channels
        if not isinstance(sigma, (list, tuple)):
            sigma = [sigma] * (image.ndim - 1)
        if len(sigma) == image.ndim - 1:
            sigma = list(sigma)
            sigma.insert(channel_axis % image.ndim, 0)
    assert image.dtype == np.float64, image.dtype
    if (out is not None) and (not np.issubdtype(out.dtype, np.floating)):
        raise ValueError(f"dtype of `out` must be float; got {out.dtype!r}.")
    return ndi.gaussian_filter(
        image, sigma, output=out, mode=mode, cval=cval, truncate=truncate
    )

def focus_corection(ang_vel, lin_vel, f, c, b):
    # speed check
    window_size = 5
    use_real_minimum = True
    approximate_out_of_image = True

    # init
    dw = np.zeros(3)
    focus_correction_valid = True

    # check if we can approximante ang vel along one axis to zero
    ang_low_idx = np.argmin(np.abs(ang_vel))
    # print(ang_low_idx)
    ang_low_idx = 2
    # if np.abs(ang_vel).mean() > low_ang_vel_threshold and \
    #    np.abs(ang_vel).min() / np.abs(ang_vel).max() > low_axis_threshold:
    #     focus_correction_valid = False

    if lin_vel[2] != 0:
        # check if focus of expansion is inside the image space
        u, v = np.round(f * lin_vel[0:2] / lin_vel[2])
        # print(ang_vel)
        # print('uv', [u, v])
        if not np.isfinite(u) or not np.isfinite(v):
            focus_correction_valid = False
        elif approximate_out_of_image:
            u = np.clip(u, -c[0], c[0] - 1)
            v = np.clip(v, -c[1], c[1] - 1)
        elif u < -c[0] or u > c[0] - 1 or v < -c[1] or v > c[1] - 1:
            focus_correction_valid = False
    else:
        focus_correction_valid = False

    # if looking good
    if focus_correction_valid:
        xw = [int(np.max([u + c[0] - window_size // 2, 0])),
              int(np.min([u + c[0] + window_size // 2 + 1, 2 * c[0]]))]
        yw = [int(np.max([v + c[1] - window_size // 2, 0])),
              int(np.min([v + c[1] + window_size // 2 + 1, 2 * c[1]]))]

        jw = np.array([[u * v / f, -(f ** 2 + u ** 2) / f, v],
                       [(f ** 2 + v ** 2) / f, -u * v / f, -u]])[None]

        jw = np.delete(jw, ang_low_idx, axis=2)
        jw = jw.reshape(jw.shape[0] * 2, 2)

        tg = -b[:, yw[0]:yw[1], xw[0]:xw[1]]
        tg = tg.reshape(-1, tg.shape[1] * tg.shape[2]).mean(axis=1)

        if use_real_minimum:
            norm = np.sqrt((b ** 2).sum(axis=0))
            # idx = np.argmin(norm)
            idx = np.argmin(gaussian(norm, sigma=2))
            yf, xf = np.unravel_index(idx, norm.shape)
            tg += b[:, yf, xf]

        try:
            dw = np.linalg.lstsq(jw, tg, rcond=None)[0]
            dw = np.insert(dw, ang_low_idx, 0, axis=0)
        except np.linalg.LinAlgError:
            pass

    return dw

def cosine_correction_torch(b, A, Jw, initial_delta=None, b_ind=None):
    n_optim_steps = 50
    lr = 1e-4
    stop_cond = 1e-12

    if initial_delta is not None:
        dw = initial_delta
    else:
        dw = np.zeros(3)

    A = A.transpose((1, 2, 0)).reshape(A.shape[1] * A.shape[2], 2)
    b = b.transpose((1, 2, 0)).reshape(b.shape[1] * b.shape[2], 2)
    Jw = Jw.transpose((2, 3, 0, 1)).reshape(Jw.shape[2] * Jw.shape[3], 2, 3)

    # import os
    # import imageio
    # A_norm = np.linalg.norm(A.reshape((960, 540, 2)), axis=2)
    # b_norm = np.linalg.norm(b.reshape((960, 540, 2)), axis=2)
    # imageio.imwrite(os.path.join('/Date2/hpc/datasets/icra21/dump', str(b_ind) + '_A_norm' + '.jpg'), A_norm)
    # imageio.imwrite(os.path.join('/Date2/hpc/datasets/icra21/dump', str(b_ind) + '_b_norm' + '.jpg'), b_norm)

    Au, Av = A[:, :1], A[:, 1:]
    bu, bv = b[:, :1], b[:, 1:]
    Ju, Jv = Jw[:, 0], Jw[:, 1]

    Ab = (A * b).sum(axis=1)[..., None]
    nA2 = (A ** 2).sum(axis=1)[..., None]
    AJ = Au * Ju + Av * Jv
    # Abx = Au * bv - Av * bu
    # AJx = Au * Jv - Av * Ju
    nb = np.sqrt(bu ** 2 + bv ** 2)

    dw = torch.tensor(dw)
    Ab, AJ, nA2 = torch.tensor(Ab), torch.tensor(AJ), torch.tensor(nA2)
    # Abx, AJx = torch.tensor(Abx), torch.tensor(AJx)
    bu, bv = torch.tensor(bu), torch.tensor(bv)
    Ju, Jv = torch.tensor(Ju), torch.tensor(Jv)
    nb = torch.tensor(nb)

    dw.requires_grad_()
    optimizer = torch.optim.SGD([dw], lr, momentum=0.2)

    def sim_loss(w):
        w = w[..., None]
        weight = nb
        weight[nb < torch.median(nb)] = 0
        err = (1 - (Ab + AJ @ w) / (torch.sqrt(nA2) * torch.sqrt((bu + Ju @ w) ** 2 + (bv + Jv @ w) ** 2))) * weight
        cos = err.mean()
        # cos = ((1 - (Ab + AJ @ w) / (torch.sqrt(nA2) * torch.sqrt((bu + Ju @ w) ** 2 + (bv + Jv @ w) ** 2))) * (nb / torch.sqrt(nA2))).mean()
        # sin = ((Abx + AJx @ w) / (torch.sqrt(nA2) * torch.sqrt((bu + Ju @ w) ** 2 + (bv + Jv @ w) ** 2))).abs().mean()
        return cos

    optimizer.zero_grad()
    loss = sim_loss(dw)
    # print('Step # {}, loss: {}'.format(0, loss.item()))
    loss.backward()
    optimizer.step()
    # scheduler.step()
    ll = loss.item()

    if np.isnan(ll):
        return np.zeros(3)

    ii = 0
    for ii in range(1, n_optim_steps):
        optimizer.zero_grad()
        loss = sim_loss(dw)
        if np.abs(ll - loss.item()) < stop_cond:
            break
        # print('Step # {}, loss: {}'.format(ii, loss.item()))
        loss.backward()
        optimizer.step()
        # scheduler.step()
        ll = loss.item()
        # print(ii, loss.item(), dw.detach().numpy())
    # print(b_ind, ii)

    dw = dw.detach().numpy()

    # if np.linalg.norm(dw) < 0.02:
    #     dw = dw
    # else:
    #     dw = np.zeros(3)
        # print(b_ind, 'trigger norm', np.linalg.norm(dw))

    return dw