"""Generalized boundaries (softseg) representation"""
import torch as tr
import numpy as np
from overrides import overrides

from .gb_impl.softseg import soft_seg
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

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, RepresentationOutput] | None = None) -> RepresentationOutput:
        x = tr.from_numpy(frames).type(tr.float) / 255
        x = x.permute(0, 3, 1, 2)
        y = soft_seg(x, use_median_filtering=self.use_median_filtering, as_image=self.adjust_to_rgb,
                     max_channels=self.max_channels)
        y = y.permute(0, 2, 3, 1).cpu().numpy()
        return RepresentationOutput(output=y)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        return (repr_data.output * 255).astype(np.uint8)

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return RepresentationOutput(output=image_resize_batch(repr_data.output, height=new_size[0], width=new_size[1]))

    @overrides
    def vre_setup(self):
        pass
