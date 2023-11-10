import gdown
import pims
from pathlib import Path
from tempfile import TemporaryDirectory

from vre import VRE
from vre.representations import build_representations_from_cfg
from vre.utils import get_project_root

def setup():
    video_path = get_project_root() / "resources/testVideo.mp4"
    if not video_path.exists():
        gdown.download("https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk", str(video_path))
    return str(video_path)

def test_vre_simple_representations():
    video_path = setup()
    video = pims.Video(video_path)
    representations_dict = {"rgb": {"type": "default", "name": "rgb", "dependencies": [], "parameters": {}}}
    representations = build_representations_from_cfg(video, representations_dict)
    vre = VRE(video, representations)
    assert vre is not None
    tmp_dir = Path(TemporaryDirectory().name)
    vre(tmp_dir, start_frame=1000, end_frame=1001, export_png=True, export_npy=True)
    assert Path(f"{tmp_dir}/rgb/npy/1000.npz").exists()
    assert Path(f"{tmp_dir}/rgb/png/1000.png").exists()
