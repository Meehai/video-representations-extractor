from tempfile import TemporaryDirectory
from pathlib import Path
import time
import numpy as np
import pytest
import pims
from vre import VRE, ReprOut
from vre.utils import FakeVideo, image_resize_batch, fetch_resource
from vre.representations.rgb import RGB

def test_vre_ctor():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    with pytest.raises(AssertionError) as e:
        _ = VRE(video=video, representations={})
    assert "At least one representation must be provided" in str(e)
    vre = VRE(video=video, representations={"rgb": RGB("rgb")})
    res = vre(Path(TemporaryDirectory().name), binary_format="npz", image_format=None)
    assert len(res["run_stats"]["rgb"]) == 2, res

def test_vre_ouput_dir_exist_mode():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), frame_rate=30)

    vre = VRE(video=video, representations={"rgb": RGB("rgb")})
    _ = vre(tmp_dir := Path(TemporaryDirectory().name), binary_format="npz", image_format=None)
    creation_time1 = (tmp_dir / "rgb/npz").stat().st_ctime
    try:
        vre(Path(TemporaryDirectory().name), binary_format="npz", image_format=None)
    except AssertionError as e:
        assert "Set --output_dir_exists_mode to 'overwrite' or 'skip_computed'" in str(e)
    _ = vre(tmp_dir, binary_format="npz", image_format=None, output_dir_exists_mode="skip_computed")
    creation_time2 = (tmp_dir / "rgb/npz").stat().st_ctime
    assert creation_time1 == creation_time2

    time.sleep(0.01)
    _ = vre(tmp_dir, binary_format="npz", image_format=None, output_dir_exists_mode="overwrite")
    creation_time3 = Path((tmp_dir / "rgb/npz").absolute()).stat().st_ctime
    assert creation_time1 < creation_time3

def test_vre_ouput_shape():
    class RGBWithShape(RGB):
        def __init__(self, shape, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape = shape

        def make(self, frames: np.ndarray, dep_data: dict) -> ReprOut:
            return ReprOut(output=image_resize_batch(super().make(frames).output, *self.shape))

    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    vre = VRE(video=video, representations={"rgb": RGBWithShape((64, 64), "rgb")})

    _ = vre(tmp_dir := Path(TemporaryDirectory().name), binary_format="npz", image_format=None, output_size="video_shape")
    assert np.load(tmp_dir / "rgb/npz/0.npz")["arr_0"].shape == (128, 128, 3)

    _ = vre(tmp_dir := Path(TemporaryDirectory().name), binary_format="npz", image_format=None, output_size="native")
    assert np.load(tmp_dir / "rgb/npz/0.npz")["arr_0"].shape == (64, 64, 3)

    _ = vre(tmp_dir := Path(TemporaryDirectory().name), binary_format="npz", image_format=None, output_size=(100, 100))
    assert np.load(tmp_dir / "rgb/npz/0.npz")["arr_0"].shape == (100, 100, 3)

def test_vre_simple_representations():
    video = pims.Video(fetch_resource("test_video.mp4"))
    representations = {"rgb": RGB(name="rgb")}
    tmp_dir = Path(TemporaryDirectory().name)
    vre = VRE(video, representations)
    assert vre is not None
    vre(tmp_dir, start_frame=1000, end_frame=1001,  binary_format="npz", image_format="jpg")
    assert Path(f"{tmp_dir}/rgb/npz/1000.npz").exists()
    assert Path(f"{tmp_dir}/rgb/jpg/1000.jpg").exists()

def test_vre_metadata():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    vre = VRE(video=video, representations={"rgb": RGB("rgb")})
    temp_dir = Path(TemporaryDirectory().name)
    res = vre.run(output_dir=temp_dir, binary_format="npz", image_format=None)
    assert res["run_stats"].keys() == {"rgb"}
    assert res["runtime_args"]["frames"] == (0, 2)

if __name__ == "__main__":
    test_vre_simple_representations()
