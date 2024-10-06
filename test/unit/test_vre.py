from tempfile import TemporaryDirectory
from pathlib import Path
import time
import numpy as np
from vre import VRE
from vre.utils import FakeVideo, image_resize_batch, RepresentationOutput
from vre.representations.rgb import RGB

def test_vre_1():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    try:
        _ = VRE(video=video, representations={})
    except AssertionError as e:
        assert "At least one representation must be provided" in str(e)

def test_vre_2():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    vre = VRE(video=video, representations={"rgb": RGB("rgb", [])})
    res = vre(Path(TemporaryDirectory().name), export_npy=True, export_png=False)
    assert len(res) == 2, res

def test_vre_ouput_dir_exist_mode():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), frame_rate=30)

    vre = VRE(video=video, representations={"rgb": RGB("rgb", [])})
    _ = vre(tmp_dir := Path(TemporaryDirectory().name), export_npy=True, export_png=False)
    creation_time1 = (tmp_dir / "rgb/npy").stat().st_ctime
    try:
        vre(Path(TemporaryDirectory().name), export_npy=True, export_png=False)
    except AssertionError as e:
        assert "Set --output_dir_exists_mode to 'overwrite' or 'skip_computed'" in str(e)
    _ = vre(tmp_dir, export_npy=True, export_png=False, output_dir_exists_mode="skip_computed")
    creation_time2 = (tmp_dir / "rgb/npy").stat().st_ctime
    assert creation_time1 == creation_time2

    time.sleep(0.01)
    _ = vre(tmp_dir, export_npy=True, export_png=False, output_dir_exists_mode="overwrite")
    creation_time3 = Path((tmp_dir / "rgb/npy").absolute()).stat().st_ctime
    assert creation_time1 < creation_time3

def test_vre_ouput_shape():
    class FakeRGB(RGB):
        def __init__(self, shape, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape = shape

        def make(self, frames: np.ndarray, dep_data: dict) -> RepresentationOutput:
            return RepresentationOutput(output=image_resize_batch(super().make(frames).output, *self.shape))

    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    vre = VRE(video=video, representations={"rgb": FakeRGB((64, 64), "rgb", [])})

    _ = vre(tmp_dir := Path(TemporaryDirectory().name), export_npy=True, export_png=False, output_size="video_shape")
    assert np.load(tmp_dir / "rgb/npy/0.npz")["arr_0"].shape == (128, 128, 3)

    _ = vre(tmp_dir := Path(TemporaryDirectory().name), export_npy=True, export_png=False, output_size="native")
    assert np.load(tmp_dir / "rgb/npy/0.npz")["arr_0"].shape == (64, 64, 3)

    _ = vre(tmp_dir := Path(TemporaryDirectory().name), export_npy=True, export_png=False, output_size=(100, 100))
    assert np.load(tmp_dir / "rgb/npy/0.npz")["arr_0"].shape == (100, 100, 3)
