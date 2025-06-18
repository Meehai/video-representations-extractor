import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from natsort import natsorted
from overrides import overrides
import shutil
import numpy as np
from vre_video import VREVideo
from vre import VRE
from vre.utils import (colorize_semantic_segmentation, semantic_mapper, DiskData,
                       MemoryData, ReprOut, get_project_root, image_resize_batch)
from vre.representations import Representation, TaskMapper, NpIORepresentation

sys.path.append(str(get_project_root() / "test/vre"))
from fake_representation import FakeRepresentation

class SemaCompute(Representation, NpIORepresentation):
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
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        raise NotImplementedError(f"[{self}] compute() must be overriden. We inherit it for output_dtype/size etc.")

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

class Buildings(TaskMapper, NpIORepresentation):
    def __init__(self, name: str, dependencies: list[Representation]):
        super().__init__(name=name, dependencies=dependencies, n_channels=2)
        NpIORepresentation.__init__(self)
        self.dtype = "bool"
        self.mapping = [
            {"buildings": [0, 1, 2, 3], "others": [4, 5, 6, 7]},
            {"buildings": [0, 1, 4, 7], "others": [2, 3, 6, 5]},
        ]
        self.original_classes = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
        self.classes = ["buildings", "others"]
        self.color_map = [[255, 255, 255], [0, 0, 0]]
        self.output_dtype = "uint8"

    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        return MemoryData(np.eye(2)[disk_data.astype(int)].astype(np.float32))

    def memory_to_disk_fmt(self, memory_data: MemoryData) -> DiskData:
        return memory_data.argmax(-1).astype(bool)

    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        dep_data_converted = [semantic_mapper(x.argmax(-1), mapping, oc)
                              for x, mapping, oc in zip(dep_data, self.mapping, self.original_classes)]
        res = self.disk_to_memory_fmt(sum(dep_data_converted) > 0)
        return res

    def make_images(self, data: ReprOut) -> np.ndarray:
        return colorize_semantic_segmentation(data.output.argmax(-1), self.classes, self.color_map)

def _generate_random_data(n: int) -> Path:
    tmp_dir = Path(__file__).parent / "data" if __name__ == "__main__" else Path(TemporaryDirectory().name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    h, w = 20, 30
    for r in ["rgb", "sema1", "sema2"]:
        (tmp_dir / r / "npz").mkdir(exist_ok=False, parents=True)
    for i in range(n):
        np.savez(f"{tmp_dir}/rgb/npz/{i}.npz", np.random.randint(0, 256, size=(h, w, 3)).astype(np.uint8))
        np.savez(f"{tmp_dir}/sema1/npz/{i}.npz", np.random.randint(0, 8, size=(h, w)).astype(np.uint8))
        np.savez(f"{tmp_dir}/sema2/npz/{i}.npz", np.random.randint(0, 8, size=(h, w)).astype(np.uint8))
    return tmp_dir

def test_vre_stored_representation():
    tmp_dir = _generate_random_data(n=3)

    video_path = Path(f"{tmp_dir}/rgb/npz")
    raw_data = [np.load(f)["arr_0"] for f in natsorted(video_path.glob("*.npz"), key=lambda p: p.name)]
    video = VREVideo(np.array(raw_data, dtype=np.uint8), fps=1)

    rgb = FakeRepresentation("rgb", n_channels=3)
    sema1 = SemaCompute("sema1", classes=8, color_map=[[i, i, i] for i in range(8)], disk_data_argmax=True)
    sema2 = SemaCompute("sema2", classes=8, color_map=[[i, i, i] for i in range(8)], disk_data_argmax=True)
    representations = [rgb, Buildings("buildings", [sema1, sema2]), sema1, sema2,
                       FakeRepresentation("hsv", n_channels=3, dependencies=[rgb])]
    vre = VRE(video, representations).set_io_parameters(binary_format="npz", image_format="png")
    print(vre)

    res = vre.run(output_dir=Path(tmp_dir), frames=list(range(0, len(video))), output_dir_exists_mode="skip_computed")
    print(res)

    # assert that it works the 2nd time too and no computation is done !
    res = vre.run(output_dir=Path(tmp_dir), frames=list(range(0, len(video))), output_dir_exists_mode="skip_computed")
    print(res)

if __name__ == "__main__":
    test_vre_stored_representation()
