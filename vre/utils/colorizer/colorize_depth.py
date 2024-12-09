"""colorize_depth -- colorized depth maps"""
import numpy as np

def colorize_depth(depth_map: np.ndarray, min_max_depth: tuple[float, float] | None = None,
                   percentiles: tuple[int, int] | None = None) -> np.ndarray:
    """Colorize depth maps. Returns a [0:1] (H, W, 3) float image. Batched."""
    assert len(depth_map.shape) == 4 and depth_map.shape[-1] == 1, depth_map.shape
    assert isinstance(depth_map, np.ndarray), depth_map
    assert (min_max_depth is not None and percentiles is None) or min_max_depth is None, (min_max_depth, percentiles)
    dm_no_nan = np.nan_to_num(depth_map[..., 0], True, 0, 0, 0)
    percentiles = [0, 100] if percentiles is None else percentiles
    min_depth, max_depth = np.percentile(dm_no_nan, percentiles)
    if min_depth == max_depth:
        return (dm_no_nan[..., None] * 0).repeat(3, axis=-1)
    dm_min_max = (dm_no_nan - min_depth) / (max_depth - min_depth)
    return _spectral(dm_min_max.clip(0, 1))

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
