import gdown
import pims
from pathlib import Path
from tempfile import TemporaryDirectory

from vre import VRE
from vre.representations import build_representations_from_cfg

def setup():
    if not Path("testVideo.mp4").exists():
        gdown.download("https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk", "testVideo.mp4")


def test_vre_1():
    setup()
    video = pims.Video("testVideo.mp4")
    representations_dict = {"rgb": {"type": "default", "method": "rgb", "dependencies": [], "parameters": {}}}
    representations = build_representations_from_cfg(video, representations_dict)
    vre = VRE(video, representations)
    assert not vre is None
    tmp_dir = Path(TemporaryDirectory().name)
    vre(tmp_dir, start_frame=1000, end_frame=1001, export_png=True, export_raw=True, export_npy=True)
    assert Path(f"{tmp_dir}/rgb/npy/raw/1000.npz").exists()
    assert Path(f"{tmp_dir}/rgb/npy/720x1280/1000.npz").exists()
    assert Path(f"{tmp_dir}/rgb/png/720x1280/1000.png").exists()
