import sys
from tempfile import TemporaryDirectory
from pathlib import Path
import random
import time
import numpy as np
import pytest
from vre import VRE, ReprOut, MemoryData, FakeVideo, VREVideo, FFmpegVideo
from vre.utils import image_resize_batch, fetch_resource, get_project_root

sys.path.append(str(get_project_root() / "test/vre"))
from fake_representation import FakeRepresentation

@pytest.fixture
def video() -> VREVideo:
    if random.randint(0, 1) % 2 == 0:
        return FakeVideo(np.random.randint(0, 255, size=(5, 128, 128, 3), dtype=np.uint8), fps=30)
    else:
        return FFmpegVideo(fetch_resource("test_video.mp4"))

def test_vre_ctor(video: VREVideo):
    with pytest.raises(AssertionError) as e:
        _ = VRE(video=video, representations=[])
    assert "At least one representation must be provided" in str(e)
    vre = VRE(video=video, representations=[FakeRepresentation("rgb", n_channels=3)])
    assert vre is not None and len(vre.representations) == 1, vre

def test_vre_run(video: VREVideo):
    vre = VRE(video=video, representations=[FakeRepresentation("rgb", n_channels=3)])
    vre.set_io_parameters(binary_format="npz", image_format="not-set")
    res = vre.run(Path(TemporaryDirectory().name), frames=[0, 1, 2])
    assert len(res.metadata["runtime_args"]["frames"])
    assert res.metadata["runtime_args"]["representations"] == ["rgb"]

def test_vre_run_with_dep(video: VREVideo):
    vre = VRE(video=video, representations=[rgb := FakeRepresentation("rgb", n_channels=3),
                                            FakeRepresentation("hsv", n_channels=3, dependencies=[rgb])])
    vre.set_io_parameters(binary_format="npz", image_format="not-set")
    res = vre.run(X := Path(TemporaryDirectory().name), frames=[0, 1, 2])
    assert len(res.metadata["runtime_args"]["frames"])
    assert res.metadata["runtime_args"]["representations"] == ["rgb", "hsv"]
    res = vre.run(X, output_dir_exists_mode="skip_computed", frames=[0, 1, 2])

def test_vre_output_dir_exists_mode(video: VREVideo):
    vre = VRE(video=video, representations=[FakeRepresentation("rgb", n_channels=3)])
    vre.set_io_parameters(binary_format="npz", image_format="not-set")
    _ = vre(tmp_dir := Path(TemporaryDirectory().name), frames=[0, 1, 2])
    creation_time1 = (tmp_dir / "rgb/npz").stat().st_ctime
    try:
        vre(Path(TemporaryDirectory().name), frames=[0, 1, 2])
    except AssertionError as e:
        assert "Set --output_dir_exists_mode to 'overwrite' or 'skip_computed'" in str(e)
    _ = vre.run(tmp_dir, output_dir_exists_mode="skip_computed", frames=[0, 1, 2])
    creation_time2 = (tmp_dir / "rgb/npz").stat().st_ctime
    assert creation_time1 == creation_time2

    time.sleep(0.01)
    _ = vre.run(tmp_dir, output_dir_exists_mode="overwrite", frames=[0, 1, 2])
    creation_time3 = Path((tmp_dir / "rgb/npz").absolute()).stat().st_ctime
    assert creation_time1 < creation_time3

def test_vre_output_shape(video: VREVideo):
    class RGBWithShape(FakeRepresentation):
        def __init__(self, shape, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape = shape

        def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
            output = MemoryData(image_resize_batch(super().compute(video, ixs).output, *self.shape))
            return ReprOut(frames=video[ixs], output=output, key=ixs)

    vre = VRE(video=video, representations=[RGBWithShape((64, 64), "rgb", n_channels=3)])
    vre.set_io_parameters(output_size="video_shape", binary_format="npz", image_format="not-set")
    _ = vre.run(tmp_dir := Path(TemporaryDirectory().name), frames=[0, 1, 2])
    assert np.load(tmp_dir / "rgb/npz/0.npz")["arr_0"].shape == video.frame_shape

    vre.set_io_parameters(output_size="native", binary_format="npz", image_format="not-set")
    _ = vre.run(tmp_dir := Path(TemporaryDirectory().name), frames=[0, 1, 2])
    assert np.load(tmp_dir / "rgb/npz/0.npz")["arr_0"].shape == (64, 64, 3)

    vre.set_io_parameters(output_size=(100, 100), binary_format="npz", image_format="not-set")
    _ = vre.run(tmp_dir := Path(TemporaryDirectory().name), frames=[0, 1, 2])
    assert np.load(tmp_dir / "rgb/npz/0.npz")["arr_0"].shape == (100, 100, 3)

def test_vre_simple_representations(video: VREVideo):
    tmp_dir = Path(TemporaryDirectory().name)
    vre = VRE(video, [FakeRepresentation("rgb", n_channels=3)])
    vre.set_io_parameters(binary_format="npz", image_format="jpg")
    vre.run(tmp_dir, frames=[4])
    assert Path(f"{tmp_dir}/rgb/npz/4.npz").exists()
    assert Path(f"{tmp_dir}/rgb/jpg/4.jpg").exists()

def test_vre_dep_data_not_saved(video: VREVideo):
    reprs = [dpt := FakeRepresentation(name="dpt", dependencies=[]),
             svd := FakeRepresentation(name="normals_svd(depth_dpt)", n_channels=3, dependencies=[dpt])]
    svd.binary_format = "npz"
    tmp_dir = Path(TemporaryDirectory().name)
    VRE(video, reprs).run(tmp_dir, frames=[2])
    assert not (tmp_dir / "dpt").exists()

def test_vre_repr_not_skipped_if_different_formats(video: VREVideo):
    tmp_dir = Path(TemporaryDirectory().name)
    rgb = FakeRepresentation("rgb", n_channels=3)
    rgb.set_io_params(binary_format="npz", image_format="not-set")

    vre = VRE(video=video, representations=[rgb])
    vre.run(tmp_dir, frames=[0, 1, 2])
    assert Path(f"{tmp_dir}/rgb/npz/0.npz").exists()
    assert not Path(f"{tmp_dir}/rgb/jpg/0.jpg").exists()

    vre.set_io_parameters(binary_format="not-set", image_format="jpg")
    vre.run(tmp_dir, frames=[2, 3])
    assert Path(f"{tmp_dir}/rgb/npz/2.npz").exists()
    assert Path(f"{tmp_dir}/rgb/jpg/2.jpg").exists()
    assert not Path(f"{tmp_dir}/rgb/npz/3.npz").exists()
    assert Path(f"{tmp_dir}/rgb/jpg/3.jpg").exists()

if __name__ == "__main__":
    test_vre_output_dir_exists_mode()
