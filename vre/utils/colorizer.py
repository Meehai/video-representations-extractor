"""common visualizing functions for standard representations, like depth, semantic, optical flow etc. Batched!"""
import numpy as np
from .colorize_sem_seg import colorize_sem_seg

def colorize_depth(depth_map: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Colorize depth maps. Returns a [0:1] (H, W, 3) float image. Batched."""
    depth_map = depth_map[..., 0:3] if depth_map.shape[-1] == 4 else depth_map
    assert len(depth_map.shape) == 3, depth_map.shape
    assert isinstance(depth_map, np.ndarray), depth_map
    depth = ((depth_map - min_depth) / (max_depth - min_depth)).clip(0, 1)
    return _spectral(depth)

def colorize_optical_flow(flow_map: np.ndarray) -> np.ndarray:
    """Colorize optical flow maps. returns a [0:255] (H, W, 3) uint8 image. Batched"""
    assert len(flow_map.shape) == 4 and flow_map.shape[-1] == 2, flow_map.shape
    return np.array([_flow_to_color(_pred) for _pred in flow_map])

def colorize_semantic_segmentation(semantic_map: np.ndarray, classes: list[str], color_map: list[tuple[int, int, int]],
                                   rgb: np.ndarray | None = None, alpha: float = 0.8):
    """Colorize asemantic segmentation maps. Must be argmaxed (H, W). Can paint over the original RGB frame or not."""
    assert np.issubdtype(semantic_map.dtype, np.integer), semantic_map.dtype
    assert (max_class := semantic_map.max()) <= len(color_map), (max_class, len(color_map))
    assert len(shp := semantic_map.shape) == 3, shp
    assert rgb is None or (rgb.shape[0:-1] == shp), (rgb.shape, shp)
    alpha = alpha if rgb is not None else 1
    rgb = rgb if rgb is not None else np.zeros((*semantic_map.shape, 3), dtype=np.uint8)
    return np.array([colorize_sem_seg(_s, _r, classes, color_map, alpha) for _r, _s in zip(rgb, semantic_map)])

# pylint: disable=invalid-name
def _make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def _flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = _make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = rad <= 1
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def _flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return _flow_uv_to_colors(u, v, convert_to_bgr)


def _spectral(grayscale_array: np.ndarray):
    """
    Maps a grayscale array (values between 0 and 1) to the spectral colormap without using Matplotlib.
    Args:
        grayscale_array (np.ndarray): Input array with values in the range [0, 1].
    Returns:
        np.ndarray: RGB array with values in the range [0, 1], shape (H, W, 3).
    """
    assert np.issubsctype(grayscale_array, np.floating), grayscale_array.dtype
    assert (_min := grayscale_array.min()) >= 0 and grayscale_array.max() <= 1, (_min, grayscale_array.max())
    spectral_colors = np.array([
        [158/255,   1/255,  66/255],  # Dark Red
        [213/255,  62/255,  79/255],  # Red
        [244/255, 109/255,  67/255],  # Orange
        [253/255, 174/255,  97/255],  # Light Orange
        [254/255, 224/255, 139/255],  # Yellow
        [230/255, 245/255, 152/255],  # Light Green
        [171/255, 221/255, 164/255],  # Green
        [102/255, 194/255, 165/255],  # Cyan
        [ 50/255, 136/255, 189/255],  # Blue
        [ 94/255,  79/255, 162/255]   # Purple
    ])

    # Corresponding grayscale positions for the colormap
    spectral_positions = np.linspace(0, 1, len(spectral_colors))

    # Interpolate each color channel separately
    red = np.interp(grayscale_array, spectral_positions, spectral_colors[:, 0])
    green = np.interp(grayscale_array, spectral_positions, spectral_colors[:, 1])
    blue = np.interp(grayscale_array, spectral_positions, spectral_colors[:, 2])

    # Stack the interpolated channels to form the RGB array
    rgb_array = np.stack([red, green, blue], axis=-1)
    return rgb_array.astype(np.float32)
