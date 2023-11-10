import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
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

def sample_representations(all_representations_dict: dict, n: int) -> dict:
    chosen_ones = np.random.choice(list(all_representations_dict.keys()), size=2, replace=False).tolist()
    representations_dict = {k: all_representations_dict[k] for k in chosen_ones}
    while True:
        deps = set()
        for v in representations_dict.values():
            for _dep in v["dependencies"]:
                if _dep not in representations_dict:
                    deps.add(_dep)
        if len(deps) == 0:
            break
        for dep in deps:
            representations_dict[dep] = all_representations_dict[dep]
    return representations_dict

def test_vre_batched():
    video_path = setup()
    video = pims.Video(video_path)
    device = "cuda" if tr.cuda.is_available() else "cpu"
    all_representations_dict = {
        "rgb": {"type": "default", "name": "rgb", "dependencies": [], "parameters": {}},
        "hsv": {"type": "default", "name": "hsv", "dependencies": [], "parameters": {}},
        "dexined": {"type": "edges", "name": "dexined", "dependencies": [],
                    "parameters": {"inference_width": 512, "inference_height": 512},
                    "vre_parameters": {"device": device}},
        "softseg gb": {"type": "soft-segmentation", "name": "generalized_boundaries", "dependencies": [],
                       "parameters": {"use_median_filtering": True, "adjust_to_rgb": True, "max_channels": 3}},
        "softseg kmeans": {"type": "soft-segmentation", "name": "kmeans", "dependencies": [],
                           "parameters": {"n_clusters": 6, "epsilon": 2, "max_iterations": 10, "attempts": 3}},
        "canny": {"type": "edges", "name": "canny", "dependencies": [],
                  "parameters": {"threshold1": 100, "threshold2": 200, "aperture_size": 3, "l2_gradient": True}},
        "depth dpt": {"type": "depth", "name": "dpt", "dependencies": [], "parameters": {},
                      "vre_parameters": {"device": device}},
        "normals svd (dpth)": {"type": "normals", "name": "depth-svd", "dependencies": ["depth dpt"],
                               "parameters": {"sensor_fov": 75, "sensor_width": 3840,
                                              "sensor_height": 2160, "window_size": 11}},
        "opticalflow rife": {"type": "optical-flow", "name": "rife", "dependencies": [],
                             "parameters": {"compute_backward_flow": False, "uhd": False},
                             "vre_parameters": {"device": device}},
        "semantic safeuav torch": {"type": "semantic_segmentation", "name": "safeuav", "dependencies": [],
                                   "parameters": {"train_height": 240, "train_width": 428, "num_classes": 8,
                                                  "color_map": [[0, 255, 0], [0, 127, 0], [255, 255, 0],
                                                                [255, 255, 255], [255, 0, 0], [0, 0, 255],
                                                                [0, 255, 255], [127, 127, 63]]}},
        "halftone": {"type": "soft-segmentation", "name": "python-halftone", "dependencies": [],
                     "parameters": {"sample": 3, "scale": 1, "percentage": 91, "angles": [0, 15, 30, 45],
                                    "antialias": False, "resolution": [240, 426]}},
        "opticalflow raft": {"type": "optical-flow", "name": "raft", "dependencies": [],
                             "parameters": {"inference_height": 360, "inference_width": 640,
                                            "small": False, "mixed_precision": False, "iters": 20},
                             "vre_parameters": {"device": device}},
        "depth odoflow (raft)": {"type": "depth", "name": "odo-flow", "dependencies": ["opticalflow raft"],
                                 "parameters": {"linear_ang_vel_correction": True, "focus_correction": True,
                                                "cosine_correction_scipy": False, "cosine_correction_gd": True,
                                                "sensor_fov": 75, "sensor_width": 3840, "sensor_height": 2160,
                                                "min_depth_meters": 0, "max_depth_meters": 400}},
    }
    # we'll just pick 2 random representations to test here
    representations_dict = sample_representations(all_representations_dict, n=2)
    representations = build_representations_from_cfg(video, representations_dict)
    representations2 = build_representations_from_cfg(video, representations_dict)

    tmp_dir = Path("here1" if __name__ == "__main__" else TemporaryDirectory().name)
    tmp_dir2 = Path("here2" if __name__ == "__main__" else TemporaryDirectory().name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(tmp_dir2, ignore_errors=True)

    start_frame, end_frame = 1000, (1100 if __name__ == "__main__" else 1005)
    batch_size = 5
    vre = VRE(video, representations)
    took1 = vre(tmp_dir, start_frame=start_frame, end_frame=end_frame, export_npy=True, export_png=True, batch_size=1)
    vre2 = VRE(video, representations2)
    took2 = vre2(tmp_dir2, start_frame=start_frame, end_frame=end_frame, export_npy=True, export_png=True,
                 batch_size=batch_size)

    both = pd.concat([pd.DataFrame(took1).drop(columns=["frame"]).mean().rename("unbatched"),
                      pd.DataFrame(took2).drop(columns=["frame"]).mean().rename(f"batch={batch_size}")], axis=1)
    both.loc["total"] = both.sum() * (end_frame - start_frame)

    for representation in vre.representations.keys():
        for t in range(start_frame, end_frame):
            a = np.load(tmp_dir / representation / "npy/" / f"{t}.npz")["arr_0"]
            b = np.load(tmp_dir2 / representation / "npy/" / f"{t}.npz")["arr_0"]
            # cannot make these ones reproductible :/
            if representation in ("softseg kmeans", ) or "odoflow" in representation:
                continue
            assert np.abs(a - b).mean() < 1e-2, (representation, t)

if __name__ == "__main__":
    test_vre_batched()
