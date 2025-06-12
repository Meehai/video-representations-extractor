import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from vre import VRE, FakeVideo
from vre.vre_runtime_args import VRERuntimeArgs
from vre.utils import get_project_root
sys.path.append(str(get_project_root() / "test/vre"))
from fake_representation import FakeRepresentation

def test_RunMetadata_1():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), fps=30)
    vre = VRE(video=video, representations=[FakeRepresentation("rgb", n_channels=3)])
    vre.set_io_parameters(binary_format="npz", image_format="not-set", compress=True, output_size=(128, 128))
    res = vre.run(output_dir := Path(TemporaryDirectory().name))
    assert res.repr_names == ["rgb"], res.repr_names
    assert res.runtime_args["frames"] == [0, 1], res.runtime_args["frames"]
    assert (output_dir / "rgb/.repr_metadata.json").exists()

def test_RepresentationMetadata_1():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), fps=30)
    rgb = FakeRepresentation("rgb", n_channels=3)
    rgb.set_io_params(binary_format="npz", image_format="not-set", compress=True, output_size=(128, 128))
    vre = VRE(video=video, representations=[rgb])

    (tmp_dir := Path(TemporaryDirectory().name)).mkdir(exist_ok=False)
    runtime_args = VRERuntimeArgs(video, [rgb], frames=None, exception_mode="stop_execution", n_threads_data_storer=0)
    meta = vre.do_one_representation(run_id="run_id", representation=rgb, output_dir=tmp_dir,
                                     output_dir_exists_mode="raise", runtime_args=runtime_args)
    assert meta.repr_name == "rgb"
    assert len(meta.run_stats) == 2 and meta.run_stats.keys() == {0, 1}, meta.run_stats

def test_load_RepresentationMetadata_1():
    # This test does a video in 2 steps. Step 1 does first 5 frames, Step 2 does the last 5 frames.
    video = FakeVideo(np.random.randint(0, 255, size=(10, 128, 128, 3), dtype=np.uint8), fps=30)
    rgb = FakeRepresentation("rgb", n_channels=3)
    rgb.set_io_params(binary_format="npz", image_format="not-set", compress=True, output_size=(128, 128))
    rgb.set_compute_params(batch_size=1)
    vre = VRE(video=video, representations=[rgb])

    (tmp_dir := Path(TemporaryDirectory().name)).mkdir(exist_ok=False)
    runtime_args = VRERuntimeArgs(video, [rgb], frames=[0, 1, 2, 3, 4],
                                  exception_mode="stop_execution", n_threads_data_storer=0)
    meta = vre.do_one_representation(run_id="run_id", representation=rgb, output_dir=tmp_dir,
                                     output_dir_exists_mode="skip_computed", runtime_args=runtime_args)
    assert meta.repr_name == "rgb"
    assert len(meta.run_stats) == 10 and meta.run_stats.keys() == set(range(10)), meta.run_stats
    assert len([v for v in meta.run_stats.values() if v is not None]) == 5

    runtime_args2 = VRERuntimeArgs(video, [rgb], frames=list(range(10)),
                                   exception_mode="stop_execution", n_threads_data_storer=0)
    meta2 = vre.do_one_representation(run_id="run_id", representation=rgb, output_dir=tmp_dir,
                                      output_dir_exists_mode="skip_computed", runtime_args=runtime_args2)
    assert len(meta2.run_stats) == 10 and meta2.run_stats.keys() == set(range(10)), meta2.run_stats
    assert len([v for v in meta2.run_stats.values() if v is not None]) == 10

    # Make sure that the metadata actually reloads the existing one and doesn't just write over it
    disk_data = json.load(open(tmp_dir / "rgb/.repr_metadata.json", "r"))
    for frame in [0, 1, 2, 3, 4]:
        assert meta2.run_stats[frame] == meta.run_stats[frame]
        assert meta2.run_stats[frame]._asdict() == disk_data["run_stats"][str(frame)] # note json conversion

def test_RunMetadata_exported_representations():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), fps=30)
    representations=[FakeRepresentation("rgb", n_channels=3), FakeRepresentation("hsv", n_channels=3)]
    vre = VRE(video=video, representations=representations)
    vre.set_io_parameters(binary_format="npz", image_format="not-set", compress=True, output_size=(128, 128))
    res = vre.run(output_dir := Path(TemporaryDirectory().name), exported_representations=["rgb"])
    assert res.repr_names == ["rgb"], res.repr_names
    assert res.runtime_args["frames"] == [0, 1], res.runtime_args["frames"]
    assert (output_dir / "rgb/.repr_metadata.json").exists()
    assert not (output_dir / "hsv/.repr_metadata.json").exists()
