import sys
from tempfile import TemporaryDirectory
from pathlib import Path
import time
import numpy as np
import pytest
from vre import VRE, ReprOut, MemoryData, FakeVideo, VREVideo, FFmpegVideo
from vre.utils import image_resize_batch, fetch_resource, get_project_root

sys.path.append(str(get_project_root() / "test/vre"))
from fake_representation import FakeRepresentation

@pytest.fixture
def video() -> FakeVideo:
    return FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), fps=30)

def test_vre_ctor(video: FakeVideo):
    with pytest.raises(AssertionError) as e:
        _ = VRE(video=video, representations=[])
    assert "At least one representation must be provided" in str(e)
    vre = VRE(video=video, representations=[FakeRepresentation("rgb", n_channels=3)])
    assert vre is not None and len(vre.representations) == 1, vre

def test_vre_run(video: FakeVideo):
    vre = VRE(video=video, representations=[FakeRepresentation("rgb", n_channels=3)])
    vre.set_io_parameters(binary_format="npz", image_format="not-set")
    res = vre.run(Path(TemporaryDirectory().name))
    assert len(res.metadata["runtime_args"]["frames"])
    assert res.metadata["runtime_args"]["representations"] == ["rgb"]

def test_vre_run_with_dep(video: FakeVideo):
    vre = VRE(video=video, representations=[rgb := FakeRepresentation("rgb", n_channels=3),
                                            FakeRepresentation("hsv", n_channels=3, dependencies=[rgb])])
    vre.set_io_parameters(binary_format="npz", image_format="not-set")
    res = vre.run(X := Path(TemporaryDirectory().name))
    assert len(res.metadata["runtime_args"]["frames"])
    assert res.metadata["runtime_args"]["representations"] == ["rgb", "hsv"]
    res = vre.run(X, output_dir_exists_mode="skip_computed")

def test_vre_output_dir_exists_mode():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), fps=30)
    vre = VRE(video=video, representations=[FakeRepresentation("rgb", n_channels=3)])
    vre.set_io_parameters(binary_format="npz", image_format="not-set")
    _ = vre(tmp_dir := Path(TemporaryDirectory().name))
    creation_time1 = (tmp_dir / "rgb/npz").stat().st_ctime
    try:
        vre(Path(TemporaryDirectory().name))
    except AssertionError as e:
        assert "Set --output_dir_exists_mode to 'overwrite' or 'skip_computed'" in str(e)
    _ = vre(tmp_dir, output_dir_exists_mode="skip_computed")
    creation_time2 = (tmp_dir / "rgb/npz").stat().st_ctime
    assert creation_time1 == creation_time2

    time.sleep(0.01)
    _ = vre(tmp_dir, output_dir_exists_mode="overwrite")
    creation_time3 = Path((tmp_dir / "rgb/npz").absolute()).stat().st_ctime
    assert creation_time1 < creation_time3

def test_vre_output_shape():
    class RGBWithShape(FakeRepresentation):
        def __init__(self, shape, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape = shape

        def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
            output = MemoryData(image_resize_batch(super().compute(video, ixs).output, *self.shape))
            return ReprOut(frames=video[ixs], output=output, key=ixs)

    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), fps=30)
    vre = VRE(video=video, representations=[RGBWithShape((64, 64), "rgb", n_channels=3)])

    vre.set_io_parameters(output_size="video_shape", binary_format="npz", image_format="not-set")
    _ = vre.run(tmp_dir := Path(TemporaryDirectory().name))
    assert np.load(tmp_dir / "rgb/npz/0.npz")["arr_0"].shape == (128, 128, 3)

    vre.set_io_parameters(output_size="native", binary_format="npz", image_format="not-set")
    _ = vre.run(tmp_dir := Path(TemporaryDirectory().name))
    assert np.load(tmp_dir / "rgb/npz/0.npz")["arr_0"].shape == (64, 64, 3)

    vre.set_io_parameters(output_size=(100, 100), binary_format="npz", image_format="not-set")
    _ = vre.run(tmp_dir := Path(TemporaryDirectory().name))
    assert np.load(tmp_dir / "rgb/npz/0.npz")["arr_0"].shape == (100, 100, 3)

def test_vre_simple_representations():
    video = FFmpegVideo(fetch_resource("test_video.mp4"))
    tmp_dir = Path(TemporaryDirectory().name)
    vre = VRE(video, [FakeRepresentation("rgb", n_channels=3)])
    vre.set_io_parameters(binary_format="npz", image_format="jpg")
    vre(tmp_dir, frames=[1000])
    assert Path(f"{tmp_dir}/rgb/npz/1000.npz").exists()
    assert Path(f"{tmp_dir}/rgb/jpg/1000.jpg").exists()

def test_vre_dep_data_not_saved():
    video = FFmpegVideo(fetch_resource("test_video.mp4"))
    reprs = [dpt := FakeRepresentation(name="dpt", dependencies=[]),
             svd := FakeRepresentation(name="normals_svd(depth_dpt)", n_channels=3, dependencies=[dpt])]
    svd.binary_format = "npz"
    tmp_dir = Path(TemporaryDirectory().name)
    VRE(video, reprs).run(tmp_dir, frames=[1000])
    assert not (tmp_dir / "dpt").exists()

if __name__ == "__main__":
    test_vre_output_dir_exists_mode()
