# pylint: disable=all
import math
import numpy as np


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
    return k.transpose()


def fov_diag_crop_aspect(fov, ar, ar_new):
    """
    Camera fov on changing aspect ratio IFF the full image senzor is used (either in height or width).
    :param fov: Diagonal FOV in degrees
    :param ar: Original aspect ratio
    :param ar_new: New aspect ratio
    :return: New FOV
    """
    dd = diag(ar / ar_new, ar) / diag(ar, 1)
    return 2 * math.atan(dd * math.tan(fov / 2 * math.pi / 180)) * 180 / math.pi


def fov_hw_to_intrinsic(fov, res):
    """
    Ideal camera matrix for horizontal / vertical fov and image size.
    :param fov: (horizontal_fov, vertical_fov) tuple
    :param res: (height, width) tuple
    :return: Camera matrix
    """
    fx = res[0] / (2 * math.tan(1 / 2 * fov[0] * math.pi / 180))
    fy = res[1] / (2 * math.tan(1 / 2 * fov[1] * math.pi / 180))
    k = np.asarray([[fx, 0, 0], [0, fy, 0], [res[0]/2, res[1]/2, 1]])
    return k.transpose()


def fov_hw_to_focal(fov, res):
    """
    fx, fy for horizontal / vertical fov and image size.
    :param fov: (horizontal_fov, vertical_fov) tuple
    :param res: (height, width) tuple
    :return: fx, fy camera parameters
    """
    fx = res[0] / (2 * math.tan(1 / 2 * fov[0] * math.pi / 180))
    fy = res[1] / (2 * math.tan(1 / 2 * fov[1] * math.pi / 180))
    return fx, fy


def focal_from_35mm(f_equiv, crop_f):
    """
    Focal length from 35mm equivalent.
    :param f_equiv: 35mm equivalent length
    :param crop_f: Senzor crop factor
    :return:
    """
    return f_equiv / crop_f


def diag(x, y):
    return math.sqrt(x ** 2 + y ** 2)


