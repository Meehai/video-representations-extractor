"""semantic_representation.py -- helper class for all semantic segmentation representations"""
from overrides import overrides
import numpy as np
from vre.representations import Representation, NpIORepresentation, ComputeRepresentationMixin
from vre.utils import DiskData, MemoryData, image_resize_batch, colorize_semantic_segmentation, ReprOut, VREVideo

class SemanticRepresentation(Representation, NpIORepresentation, ComputeRepresentationMixin):
    """SemanticRepresentation. Implements semantic task-specific stuff, like argmaxing if needed"""
    def __init__(self, *args, classes: int | list[str], color_map: list[tuple[int, int, int]],
                 semantic_argmax_only: bool = False, **kwargs):
        self.n_classes = len(list(range(classes)) if isinstance(classes, int) else classes)
        Representation.__init__(self, *args, **kwargs)
        NpIORepresentation.__init__(self)
        ComputeRepresentationMixin.__init__(self)
        self.classes = list(range(classes)) if isinstance(classes, int) else classes
        self.color_map = color_map
        self.semantic_argmax_only = semantic_argmax_only
        assert len(color_map) == self.n_classes and self.n_classes > 1, (color_map, self.n_classes)
        self._output_dtype = "uint8" if semantic_argmax_only else "float16"

    @property
    @overrides
    def n_channels(self) -> int:
        return self.n_classes

    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        raise NotImplementedError(f"[{self}] compute() must be overriden. We inherit it for output_dtype/size etc.")

    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        assert disk_data.dtype in (np.uint8, np.uint16), disk_data.dtype
        if self.semantic_argmax_only:
            return MemoryData(disk_data)
        return MemoryData(np.eye(len(self.classes))[disk_data].astype(np.float32))

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        assert data is not None, f"[{self}] data must be first computed using compute()"
        frames_rsz = None
        if data.frames is not None:
            frames_rsz = image_resize_batch(data.frames, *data.output.shape[1:3])
        preds = self.to_argmaxed_representation(data.output)
        return colorize_semantic_segmentation(preds, self.classes, self.color_map, rgb=frames_rsz)

    def to_argmaxed_representation(self, memory_data: MemoryData) -> MemoryData:
        """returns the argmaxed representation"""
        return memory_data if self.semantic_argmax_only else memory_data.argmax(-1)
