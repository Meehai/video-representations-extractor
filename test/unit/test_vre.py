from tempfile import TemporaryDirectory
from pathlib import Path
from vre import VRE
from vre.utils import FakeVideo
import numpy as np
import time
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
    tmp_dir = Path(TemporaryDirectory().name)
    res = vre(tmp_dir, export_npy=True, export_png=False)
    assert len(res) == 2, res

def test_vre_ouput_dir_exist_mode():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    vre = VRE(video=video, representations={"rgb": RGB("rgb", [])})
    tmp_dir = Path(TemporaryDirectory().name)
    _ = vre(tmp_dir, export_npy=True, export_png=False)
    creation_time1 = tmp_dir.stat().st_ctime
    try:
        vre(tmp_dir, export_npy=True, export_png=False)
    except AssertionError as e:
        assert "Set mode to 'overwrite' or 'skip_computed'" in str(e)
    _ = vre(tmp_dir, export_npy=True, export_png=False, output_dir_exist_mode="skip_computed")
    creation_time2 = tmp_dir.stat().st_ctime
    assert creation_time1 == creation_time2

    time.sleep(0.01)
    _ = vre(tmp_dir, export_npy=True, export_png=False, output_dir_exist_mode="overwrite")
    creation_time3 = Path(tmp_dir.absolute()).stat().st_ctime
    assert creation_time1 < creation_time3
