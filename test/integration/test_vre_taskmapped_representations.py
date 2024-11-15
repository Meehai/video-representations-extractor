from tempfile import TemporaryDirectory
from pathlib import Path
from natsort import natsorted
import shutil
import numpy as np
import pandas as pd
from vre import VRE
from vre.utils import FakeVideo, colorize_semantic_segmentation, semantic_mapper
from vre.representations import Representation, TaskMapper, NpIORepresentation, DiskData, MemoryData
from vre.representations.color import RGB, HSV
from vre.representations.cv_representations import SemanticRepresentation

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
        return np.eye(2)[disk_data.astype(int)].astype(np.float32)

    def memory_to_disk_fmt(self, memory_data: MemoryData) -> DiskData:
        return memory_data.argmax(-1).astype(bool)

    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        dep_data_converted = [semantic_mapper(x.argmax(-1), mapping, oc)
                              for x, mapping, oc in zip(dep_data, self.mapping, self.original_classes)]
        res = self.disk_to_memory_fmt(sum(dep_data_converted) > 0)
        return res

    def make_images(self) -> np.ndarray:
        res = [colorize_semantic_segmentation(item.argmax(-1).astype(int), self.classes, color_map=self.color_map,
                                              original_rgb=None, font_size_scale=2) for item in self.data.output]
        return np.array(res)

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
    video = FakeVideo(np.array(raw_data, dtype=np.uint8), fps=1)

    rgb = RGB("rgb")
    sema1 = SemanticRepresentation("sema1", classes=8, color_map=[[i, i, i] for i in range(8)])
    sema2 = SemanticRepresentation("sema2", classes=8, color_map=[[i, i, i] for i in range(8)])
    representations = {"rgb": rgb, "buildings": Buildings("buildings", [sema1, sema2]), "hsv": HSV("hsv", [rgb])}
    vre = VRE(video, representations).set_io_parameters(binary_format="npz", image_format="png")
    print(vre)

    res = vre.run(output_dir=Path(tmp_dir), frames=list(range(0, len(video))), output_dir_exists_mode="skip_computed")
    vre_run_stats = pd.DataFrame(res["run_stats"], index=res["runtime_args"]["frames"])
    print(vre_run_stats)

    # assert that it works the 2nd time too and no computation is done !
    res = vre.run(output_dir=Path(tmp_dir), frames=list(range(0, len(video))), output_dir_exists_mode="skip_computed")

if __name__ == "__main__":
    test_vre_stored_representation()
