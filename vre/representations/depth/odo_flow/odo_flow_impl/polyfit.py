# pylint: disable=all
import numpy as np


def poly_fit_traj_fixed_dt(dt, x, window=51, degree=2):
    assert (window % 2)
    assert (degree < window)

    t = np.arange(-(window // 2), (window // 2) + 1) * dt
    A = build_design_matrix(t, degree)
    C = np.linalg.inv(A.T @ A) @ A.T

    data_per_window = np.full((len(x), window), np.nan)
    mask = np.full(len(x), True, dtype=bool)
    for ind in range(len(x)):
        start = ind - window // 2
        end = ind + window // 2
        if start < 0 or end >= len(x):
            mask[ind] = False
            continue
        data_per_window[ind] = x[start:end + 1]

    bs = data_per_window
    # solve least squares
    poly_per_window = np.squeeze(C @ np.expand_dims(bs, axis=2), axis=2)

    return poly_per_window, mask


def poly_fit_traj(t, x, window=51, degree=2, poly_fit='normal_eq', center=False):
    assert (window % 2)
    assert (degree < window)

    data_per_window = np.full((len(x), window), np.nan)
    time_per_window = np.full((len(x), window), np.nan)
    mask = np.full(len(x), True, dtype=bool)
    for ind in range(len(x)):
        start = ind - window // 2
        end = ind + window // 2
        if start < 0 or end >= len(x):
            mask[ind] = False
            continue
        data_per_window[ind] = x[start:end + 1]
        time_per_window[ind] = t[start:end + 1]

    poly_per_window = np.full((len(x), degree + 1), np.nan)

    for ind in range(len(x)):
        if not mask[ind]:
            continue
        win_x, win_t = data_per_window[ind], time_per_window[ind]
        if center:
            win_t -= t[ind]
        # poly = np.polyfit(win_t, win_x, deg=degree)[::-1]
        poly = poly_fit_1d(win_t, win_x, degree, poly_fit)

        if poly is None:
            mask[ind] = False
            continue
        poly_per_window[ind] = poly

    return poly_per_window, mask


def linear_least_squares(A, b, method='normal_eq'):
    try:
        if method == 'normal_eq':
            solution = np.linalg.inv(A.T @ A) @ A.T @ b
        else:
            solution = np.linalg.pinv(A).dot(b)
    except:
        return None
    return solution


def poly_fit_1d(win_t, win_x, degree, method='normal_eq'):
    A = build_design_matrix(win_t, degree)
    b = win_x
    return linear_least_squares(A, b, method)


def build_design_matrix(win_t, degree):
    A = []
    for ind in range(degree + 1):
        A.append(win_t ** ind)
    A = np.stack(A, axis=1)
    return A