import gdown
import pims
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime

from vre import VRE
from vre.representations import build_representations_from_cfg
from vre.utils import get_project_root

def setup():
    video_path = get_project_root() / "resources/testVideo.mp4"
    if not video_path.exists():
        gdown.download("https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk", str(video_path))
    return str(video_path)

def test_vre_batched():
    video_path = setup()
    video = pims.Video(video_path)
    representations_dict = {
        "rgb": {"type": "default", "method": "rgb", "dependencies": [], "parameters": {}},
        "hsv": {"type": "default", "method": "hsv", "dependencies": [], "parameters": {}},
        # "dexined": {"type": "edges", "method": "dexined", "dependencies": [], "parameters": {"device": "cuda"}},
        "softseg gb": {"type": "soft-segmentation", "method": "generalized_boundaries", "dependencies": [],
                        "parameters": {"useFiltering": True, "adjustToRGB": True, "maxChannels": 3}},
    }

    representations = build_representations_from_cfg(video, representations_dict)
    representations2 = build_representations_from_cfg(video, representations_dict)

    tmp_dir = Path("here1" if __name__ == "__main__" else TemporaryDirectory().name)
    tmp_dir2 = Path("here2" if __name__ == "__main__" else TemporaryDirectory().name)

    start_frame, end_frame = 1000, 1020
    vre = VRE(video, representations)
    took1 = vre(tmp_dir, start_frame=start_frame, end_frame=end_frame, export_raw=True, export_png=True, batch_size=1)
    vre2 = VRE(video, representations2)
    took2 = vre2(tmp_dir2, start_frame=start_frame, end_frame=end_frame, export_raw=True, export_png=True, batch_size=5)

    for representation in vre.representations.keys():
        for t in range(start_frame, end_frame):
            a = np.load(tmp_dir / representation / "npy/raw" / f"{t}.npz")["arr_0"]
            b = np.load(tmp_dir2 / representation / "npy/raw" / f"{t}.npz")["arr_0"]
            assert np.abs(a - b).mean() < 1e-2, (representation, t)

if __name__ == "__main__":
    test_vre_batched()
