"""Generalized boundaries (softseg) representation"""
import torch as tr
import numpy as np
from overrides import override
from .softseg import soft_seg
from ....representation import Representation, RepresentationOutput
from ....utils import image_resize_batch


class GeneralizedBoundaries(Representation):
    """
    Soft-seg implementation from https://link.springer.com/chapter/10.1007/978-3-642-33765-9_37
    Parameters:
    - use_median_filtering: Apply a median filtering postprocessing pass.
    - adjust_to_rgb: Return a RGB soft segmentation image in a similar colormap as the input.
    - max_channels: Max segmentation maps. Upper bounded at ~60.
    """
    def __init__(self, use_median_filtering: bool, adjust_to_rgb: bool, max_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.use_median_filtering = use_median_filtering
        self.adjust_to_rgb = adjust_to_rgb
        self.max_channels = max_channels

    @override
    def vre_setup(self, **kwargs):
        pass

    @override
    def make(self, t: slice) -> RepresentationOutput:
        x = tr.from_numpy(np.array(self.video[t])).type(tr.float) / 255
        x = x.permute(0, 3, 1, 2)
        y = soft_seg(x, use_median_filtering=self.use_median_filtering, as_image=self.adjust_to_rgb,
                     max_channels=self.max_channels)
        y = y.permute(0, 2, 3, 1).cpu().numpy()
        return y

    @override
    def make_images(self, t: slice, x: np.ndarray, extra: dict | None) -> np.ndarray:
        x_rsz = image_resize_batch(x, height=self.video.frame_shape[0], width=self.video.frame_shape[1])
        y = (x_rsz * 255).astype(np.uint8)
        return y
