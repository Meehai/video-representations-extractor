"""resizable_representation_mixin.py -- Interface that allows us to resize a ReprOut but also overwrite it"""
from abc import ABC
from vre.utils import image_resize_batch
import numpy as np
from .representation import ReprOut

class ResizableRepresentationMixin(ABC):
    """Interface that allows us to resize a ReprOut but also overwrite it"""
    def resize(self, data: ReprOut, new_size: tuple[int, int]):
        """resizes the data. size is provided in (h, w)"""
        assert data is not None, "No data provided"
        interpolation = "nearest" if np.issubdtype(d := data.output.dtype, np.integer) or d == bool else "bilinear"
        output_images = None
        if data.output_images is not None:
            output_images = image_resize_batch(data.output_images, *new_size, interpolation="nearest")
        return ReprOut(frames=data.frames, key=data.key, extra=data.extra, output_images=output_images,
                       output=image_resize_batch(data.output, *new_size, interpolation=interpolation))
