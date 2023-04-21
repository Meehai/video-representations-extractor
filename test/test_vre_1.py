import gdown
from vre import VRE
from pathlib import Path
from omegaconf import DictConfig
from media_processing_lib.video import MPLVideo, video_read


def setup():
    if not Path("testVideo.mp4").exists():
        gdown.download("https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk", "testVideo.mp4")


def test_vre_1():
    setup()
    video = MPLVideo(video_read("testVideo.mp4"))
    vre = VRE(
        video, representations_dict={"rgb": {"type": "default", "method": "rgb", "dependencies": [], "parameters": {}}}
    )
    assert not vre is None
    vre.run_cfg(
        Path("tmp/vre"),
        {"start_frame": 1000, "end_frame": 1001, "export_png": True, "export_raw": True, "export_npy": True},
    )
    assert Path("tmp/vre/rgb/npy/raw/1000.npz").exists()
    assert Path("tmp/vre/rgb/npy/720x1280/1000.npz").exists()
    assert Path("tmp/vre/rgb/png/720x1280/1000.png").exists()
