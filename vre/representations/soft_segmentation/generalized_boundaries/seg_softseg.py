from overrides import overrides
import torch as tr
import numpy as np
from .softseg import soft_seg
from ...representation import Representation, RepresentationOutput


class GeneralizedBoundaries(Representation):
    def __init__(self, useFiltering: bool, adjustToRGB: bool, maxChannels: int, **kwargs):
        """
        Soft-seg implementation from https://link.springer.com/chapter/10.1007/978-3-642-33765-9_37
        @param useFiltering: Apply a median filtering postprocessing pass.
        @param adjustToRGB: Return a RGB soft segmentation image in a similar colormap as the input.
        @param maxChannels: Max segmentation maps. Upper bounded at ~60.
        """
        super().__init__(**kwargs)
        self.useFiltering = useFiltering
        self.adjustToRGB = adjustToRGB
        self.maxChannels = maxChannels

    def make(self, t: int) -> RepresentationOutput:
        x = tr.from_numpy(self.video[t]).type(tr.float) / 255
        x = x.permute(2, 0, 1).unsqueeze(0)
        y = soft_seg(x, use_filtering=self.useFiltering, as_image=self.adjustToRGB, max_channels=self.maxChannels)
        y = y[0].permute(1, 2, 0).cpu().numpy()
        return y

    def make_image(self, x: RepresentationOutput) -> np.ndarray:
        y = np.uint8(x["data"] * 255)
        return y

    @overrides
    def setup(self):
        pass
