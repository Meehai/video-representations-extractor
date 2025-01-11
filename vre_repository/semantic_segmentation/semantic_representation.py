"""semantic_representation.py -- helper class for all semantic segmentation representations"""
from overrides import overrides
import numpy as np
from vre.representations import Representation, NpIORepresentation
from vre.utils import DiskData, MemoryData, image_resize_batch, colorize_semantic_segmentation, ReprOut

class SemanticRepresentation(Representation, NpIORepresentation):
    """SemanticRepresentation. Implements semantic task-specific stuff, like argmaxing if needed"""
    def __init__(self, *args, classes: int | list[str], color_map: list[tuple[int, int, int]],
                 disk_data_argmax: bool, **kwargs):
        self.n_classes = len(list(range(classes)) if isinstance(classes, int) else classes)
        Representation.__init__(self, *args, **kwargs)
        NpIORepresentation.__init__(self)
        self.classes = list(range(classes)) if isinstance(classes, int) else classes
        self.color_map = color_map
        self.disk_data_argmax = disk_data_argmax
        assert len(color_map) == self.n_classes and self.n_classes > 1, (color_map, self.n_classes)
        self._output_dtype = "uint8" if disk_data_argmax else "float16"

    @property
    @overrides
    def n_channels(self) -> int:
        return self.n_classes

    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        memory_data = MemoryData(disk_data)
        if self.disk_data_argmax:
            assert disk_data.dtype in (np.uint8, np.uint16), disk_data.dtype
            memory_data = MemoryData(np.eye(len(self.classes))[disk_data].astype(np.float32))
        assert memory_data.dtype == np.float32, memory_data.dtype
        return memory_data

    @overrides
    def memory_to_disk_fmt(self, memory_data: MemoryData) -> DiskData:
        assert memory_data.shape[-1] == self.n_classes, (memory_data.shape, self.n_classes)
        return memory_data.argmax(-1) if self.disk_data_argmax else memory_data

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        assert data is not None, f"[{self}] data must be first computed using compute()"
        frames_rsz = None
        if data.frames is not None:
            frames_rsz = image_resize_batch(data.frames, *data.output.shape[1:3])
        return colorize_semantic_segmentation(data.output.argmax(-1), self.classes, self.color_map, rgb=frames_rsz)
