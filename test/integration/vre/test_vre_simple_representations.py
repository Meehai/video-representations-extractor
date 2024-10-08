import pims
from pathlib import Path
from tempfile import TemporaryDirectory

from vre import VRE
from vre.representations import build_representations_from_cfg
from vre.utils import fetch_resource


def test_vre_simple_representations():
    video = pims.Video(fetch_resource("test_video.mp4"))
    representations_dict = {"rgb": {"type": "default/rgb", "dependencies": [], "parameters": {}}}
    representations = build_representations_from_cfg(representations_dict)
    tmp_dir = Path(TemporaryDirectory().name)
    vre = VRE(video, representations)
    assert vre is not None
    vre(tmp_dir, start_frame=1000, end_frame=1001, export_png=True, export_npy=True)
    assert Path(f"{tmp_dir}/rgb/npy/1000.npz").exists()
    assert Path(f"{tmp_dir}/rgb/png/1000.png").exists()

if __name__ == "__main__":
    test_vre_simple_representations()
