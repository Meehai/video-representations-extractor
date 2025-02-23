import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from vre import VRE, FakeVideo
from vre.vre_runtime_args import VRERuntimeArgs
from vre.utils import get_project_root
sys.path.append(str(get_project_root() / "test/vre"))
from fake_representation import FakeRepresentation

def test_vre_RunMetadata_1():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), fps=30)
    vre = VRE(video=video, representations=[FakeRepresentation("rgb", n_channels=3)])
    vre.set_io_parameters(binary_format="npz", image_format="not-set")
    res = vre.run(output_dir=Path(TemporaryDirectory().name))
    assert res.representations == ["rgb"], res.representations
    assert res.repr_metadatas["rgb"] is not None, res.repr_metadatas
    assert res.runtime_args["frames"] == [0, 1], res.runtime_args["frames"]

def test_vre_RepresentationMetadata_1():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), fps=30)
    rgb = FakeRepresentation("rgb", n_channels=3)
    rgb.set_io_params(binary_format="npz", image_format="not-set")
    vre = VRE(video=video, representations=[rgb])

    (tmp_dir := Path(TemporaryDirectory().name)).mkdir(exist_ok=False)
    runtime_args = VRERuntimeArgs(video, [rgb], frames=None, exception_mode="stop_execution", n_threads_data_storer=0)
    meta = vre._do_one_representation(rgb, tmp_dir, output_dir_exists_mode="raise", runtime_args=runtime_args)
    assert meta.repr_name == "rgb"
    assert len(meta.run_stats) == 2 and meta.run_stats.keys() == {"0", "1"}, meta.run_stats
