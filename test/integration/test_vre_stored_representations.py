from tempfile import TemporaryDirectory
from pathlib import Path
from natsort import natsorted
import shutil
import numpy as np
import torch as tr
import pandas as pd
from torch.nn import functional as F
from vre import VRE
from vre.utils import FakeVideo, colorize_semantic_segmentation, semantic_mapper
from vre.representations import Representation, TaskMapper
from vre.representations.color import RGB, HSV
from vre.representations.cv_representations import SemanticRepresentation

class Buildings(TaskMapper):
    def __init__(self, name: str, dependencies: list[Representation]):
        super().__init__(name=name, dependencies=dependencies, n_channels=2)
        self.dtype = "bool"
        self.mapping = [
            {"buildings": [0, 1, 2, 3], "others": [4, 5, 6, 7]},
            {"buildings": [0, 1, 4, 7], "others": [2, 3, 6, 5]},
        ]
        self.original_classes = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
        self.classes = ["buildings", "others"]

    def return_fn(self, x: tr.Tensor) -> tr.Tensor:
        return F.one_hot(x.long(), num_classes=2).float()

    def merge_fn(self, dep_data: list[np.ndarray]) -> tr.Tensor:
        dep_data_argmaxed = [x.argmax(-1).numpy() for x in dep_data]
        dep_data_converted = [semantic_mapper(x, mapping, oc)
                              for x, mapping, oc in zip(dep_data_argmaxed, self.mapping, self.original_classes)]
        combined = tr.from_numpy(sum(dep_data_converted) > 0) # mode='all_agree' in the original code
        return combined

    def plot_fn(self, x: tr.Tensor) -> np.ndarray:
        x_argmax = x.squeeze().nan_to_num(0).detach().argmax(-1).cpu().numpy()
        return colorize_semantic_segmentation(x_argmax, self.classes, [[255, 255, 255], [0, 0, 0]],
                                              original_rgb=None, font_size_scale=3)

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
    video = FakeVideo(np.array(raw_data, dtype=np.uint8), frame_rate=1)

    rgb = RGB("rgb")
    sema1 = SemanticRepresentation("sema1", classes=8, color_map=[[i, i, i] for i in range(8)])
    sema2 = SemanticRepresentation("sema2", classes=8, color_map=[[i, i, i] for i in range(8)])
    representations = {"rgb": rgb, "buildings": Buildings("buildings", [sema1, sema2]), "hsv": HSV("hsv", [rgb])}
    vre = VRE(video, representations)
    vre.set_compute_params(binary_format="npz", image_format="png")
    print(vre)

    res = vre.run(output_dir=Path(tmp_dir), start_frame=0, end_frame=None, output_dir_exists_mode="skip_computed")
    vre_run_stats = pd.DataFrame(res["run_stats"], index=range(*res["runtime_args"]["frames"]))
    print(vre_run_stats)

if __name__ == "__main__":
    test_vre_stored_representation()
