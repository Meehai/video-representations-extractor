import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime
import gdown
import pims
import pandas as pd
import numpy as np
import torch as tr

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
    device = "cuda" if tr.cuda.is_available() else "cpu"
    representations_dict = {
        "rgb": {"type": "default", "method": "rgb", "dependencies": [], "parameters": {}},
        "hsv": {"type": "default", "method": "hsv", "dependencies": [], "parameters": {}},
        "dexined": {"type": "edges", "method": "dexined", "dependencies": [], "parameters": {"device": "cuda:0"}},
        "softseg gb": {"type": "soft-segmentation", "method": "generalized_boundaries", "dependencies": [],
                        "parameters": {"useFiltering": True, "adjustToRGB": True, "maxChannels": 3}},
        "softseg kmeans": {"type": "soft-segmentation", "method": "kmeans", "dependencies": [],
                            "parameters": {"n_clusters": 6, "epsilon": 2, "max_iterations": 10, "attempts": 3}},
        "canny": {"type": "edges", "method": "canny", "dependencies": [],
                  "parameters": {"threshold1": 100, "threshold2": 200, "aperture_size": 3, "l2_gradient": True}},
        "depth dpt": {"type": "depth", "method": "dpt", "dependencies": [], "parameters": {"device": "cuda:1"}},
        "normals svd (dpth)": {"type": "normals", "method": "depth-svd", "dependencies": ["depth dpt"],
                               "parameters": {"sensor_fov": 75, "sensor_width": 3840,
                                              "sensor_height": 2160, "window_size": 11}},
        "opticalflow rife": {"type": "optical-flow", "method": "rife", "dependencies": [],
                             "parameters": {"compute_backward_flow": False, "device": "cuda:2"}},
        "semantic safeuav torch": {"type": "semantic", "method": "safeuav", "dependencies": [],
                                   "parameters": {"device": "cuda:3",
                                                  "weights_file": "safeuav_semantic_0956_pytorch.ckpt",
                                                  "train_height": 240, "train_width": 428, "num_classes": 8,
                                                  "color_map": [[0, 255, 0], [0, 127, 0], [255, 255, 0],
                                                               [255, 255, 255], [255, 0, 0], [0, 0, 255],
                                                               [0, 255, 255], [127, 127, 63]]}},
        "halftone": {"type": "soft-segmentation", "method": "python-halftone", "dependencies": [],
                        "parameters": {"sample": 3, "scale": 1, "percentage": 91, "angles": [0, 15, 30, 45],
                                        "antialias": False, "resolution": [240, 426]}},
        "opticalflow raft": {"type": "optical-flow", "method": "raft", "dependencies": [],
                             "parameters": {"device": device, "inference_height": 360, "inference_width": 640,
                                            "small": False, "mixed_precision": False, "iters": 20}},
        "depth odoflow (raft)": {"type": "depth", "method": "odo-flow", "dependencies": ["opticalflow raft"],
                                 "parameters": {"velocities_path": "DJI_0956_velocities.npz",
                                                "linearAngVelCorrection": True, "focus_correction": True,
                                                "cosine_correction_scipy": False, "cosine_correction_GD": True,
                                                "sensor_fov": 75, "sensor_width": 3840, "sensor_height": 2160,
                                                "min_depth_meters": 0, "max_depth_meters": 400}},
    }
    # we'll just pick 2 random representations to test here
    representations_dict = {k: v for k, v in representations_dict
                            if k in np.random.choice(list(representations_dict.keys()), 2)}

    representations = build_representations_from_cfg(video, representations_dict)
    representations2 = build_representations_from_cfg(video, representations_dict)

    tmp_dir = Path("here1" if __name__ == "__main__" else TemporaryDirectory().name)
    tmp_dir2 = Path("here2" if __name__ == "__main__" else TemporaryDirectory().name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(tmp_dir2, ignore_errors=True)

    start_frame, end_frame = 1000, 1010
    batch_size = 5
    vre = VRE(video, representations)
    took1 = vre(tmp_dir, start_frame=start_frame, end_frame=end_frame, export_raw=True, export_png=True, batch_size=1)
    vre2 = VRE(video, representations2)
    took2 = vre2(tmp_dir2, start_frame=start_frame, end_frame=end_frame, export_raw=True, export_png=True,
                 batch_size=batch_size)
    both = pd.concat([took1.drop(columns=["frame"]).mean().rename("unbatched"),
                      took2.drop(columns=["frame"]).mean().rename(f"batch={batch_size}")], axis=1)
    both.loc["total"] = both.sum() * (end_frame - start_frame)
    print(both)

    for representation in vre.representations.keys():
        for t in range(start_frame, end_frame):
            a = np.load(tmp_dir / representation / "npy/raw" / f"{t}.npz")["arr_0"]
            b = np.load(tmp_dir2 / representation / "npy/raw" / f"{t}.npz")["arr_0"]
            # cannot make these ones reproductible :/
            if representation in ("softseg kmeans", ) or "odoflow" in representation:
                continue
            assert np.abs(a - b).mean() < 1e-2, (representation, t)

if __name__ == "__main__":
    test_vre_batched()
