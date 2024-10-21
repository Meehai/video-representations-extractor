"""common visualizing functions for standard representations, like depth, semantic, optical flow etc."""
import matplotlib
import flow_vis
import numpy as np
from .colorized_sem_seg import ColorizerSemSeg

def colorize_depth(depth_map: np.ndarray, min_depth: float, max_depth: float, cmap="Spectral") -> np.ndarray:
    """Colorize depth maps. Returns a [0:1] (H, W, 3) float image."""
    assert len(depth_map.shape) == 3 or (len(depth_map.shape) == 4 and depth_map.shape[-1] == 1), depth_map.shape
    assert isinstance(depth_map, np.ndarray), depth_map
    cm = matplotlib.colormaps[cmap]
    depth = ((depth_map - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[..., 0:3]  # value from 0 to 1
    return img_colored_np.astype(np.float32)

def colorize_optical_flow(flow_map: np.ndarray) -> np.ndarray:
    """Colorize optical flow maps. returns a [0:255] (H, W, 3) uint8 image."""
    assert len(flow_map.shape) == 3 and flow_map.shape[2] == 2, flow_map.shape
    return flow_vis.flow_to_color(flow_map)

def colorize_semantic_segmentation(semantic_map: np.ndarray, classes: list[str], color_map: list[tuple[int, int, int]],
                                   original_rgb: np.ndarray | None, font_size_scale: float=1, alpha: float = 0.8):
    """Colorize asemantic segmentation maps. Must be argmaxed (H, W). Can paint over the original RGB frame or not."""
    assert np.issubdtype(semantic_map.dtype, np.integer), semantic_map.dtype
    assert (max_class := semantic_map.max()) <= len(color_map), (max_class, len(color_map))
    assert len(semantic_map.shape) == 2, semantic_map.shape
    rgb = original_rgb if original_rgb is not None else np.zeros((*semantic_map.shape, 3), dtype=np.uint8)
    alpha = alpha if original_rgb is not None else 1
    return ColorizerSemSeg(rgb, classes, color_map, font_size_scale, alpha).draw_sem_seg(semantic_map).get_image()
