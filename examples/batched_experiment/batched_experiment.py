#!/usr/bin/env python3
"""experiment using batches of frames"""
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import partial
import gdown
import pims
import pandas as pd
import torch as tr
import os

from vre import VRE
from vre.logger import logger
from vre.representations import build_representations_from_cfg
from vre.utils import get_project_root

def dwd_video_if_needed() -> str:
    """download the video in the resources dir if not exist and return the path"""
    video_path = get_project_root() / "resources/testVideo.mp4"
    if not video_path.exists():
        video_path.parent.mkdir(exist_ok=True, parents=True)
        gdown.download("https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk", f"{video_path}")
    return f"{video_path}"

def get_representation_dict() -> dict:
    """setup all representations we want to use, including one representation per device"""
    device = "cuda" if tr.cuda.is_available() else "cpu"
    all_representations_dict: dict[str, dict] = {
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
                                                                [0, 255, 255], [127, 127, 63]]},
                                   "vre_parameters": {"device": device,
                                                      "weights_file": "safeuav_semantic_0956_pytorch.ckpt"}},
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
                                                "min_depth_meters": 0, "max_depth_meters": 400},
                                 "vre_parameters": {"velocities_path": "DJI_0956_velocities.npz"}},
        "fastsam (s)": {"type": "semantic_segmentation", "name": "fastsam", "dependencies": [],
                        "parameters": {"variant": "fastsam-s", "iou": 0.9, "conf": 0.4},
                        "vre_parameters": {"device": device}},
        "fastsam (x)": {"type": "semantic_segmentation", "name": "fastsam", "dependencies": [],
                        "parameters": {"variant": "fastsam-s", "iou": 0.9, "conf": 0.4},
                        "vre_parameters": {"device": device}},
        "mask2former (r50)": {"type": "semantic_segmentation", "name": "mask2former", "dependencies": [],
                              "batch_size": 2, "vre_parameters": {"device": device},
                              "parameters": {"model_id": "49189528_1", "semantic": True, "instance": False,
                                             "panoptic": False, "semantic_argmax_only": False}},
        "mask2former (swin)": {"type": "semantic_segmentation", "name": "mask2former", "dependencies": [],
                               "batch_size": 2, "vre_parameters": {"device": device},
                               "parameters": {"model_id": "47429163_0", "semantic": True, "instance": False,
                                              "panoptic": False, "semantic_argmax_only": True}},
    }

    if os.getenv("ONLY_RGB", "0") == "1":
        all_representations_dict = {"rgb": all_representations_dict["rgb"]}

    if not tr.cuda.is_available():
        logger.info("Using CPU")
        return all_representations_dict
    if tr.cuda.device_count() == 1:
        logger.info("Using 1 GPU")
        return all_representations_dict
    n_needed = 0
    for v in all_representations_dict.values():
        if "device" in v.get("vre_parameters", {}):
            n_needed += 1
    if n_needed > tr.cuda.device_count():
        logger.info(f"Using 1 gpu. n_needed={n_needed}, n_available={tr.cuda.device_count()}")
        return all_representations_dict
    logger.info(f"Using {tr.cuda.device_count()} GPUs")
    i = 0
    for k in all_representations_dict.keys():
        if "device" in all_representations_dict[k].get("vre_parameters", {}):
            all_representations_dict[k]["vre_parameters"]["device"] = f"cuda:{i}"
            i += 1
    return all_representations_dict

def _process_dict(data: dict, batch_size: int) -> pd.Series:
    return pd.DataFrame(data).mean().rename(f"batch={batch_size}") / batch_size

def _process_all(results: list[dict], batch_sizes: list[int]) -> pd.DataFrame:
    res_mean = [_process_dict(d, bs) for d, bs in zip(results, batch_sizes)]
    res = pd.concat(res_mean, axis=1)
    if "batch=1" in res.columns:
        other_cols = sorted([col for col in res.columns if col != "batch=1"])
        new_order = []
        for col in other_cols:
            res[f"ratio1/{col.split('=')[-1]}"] = res["batch=1"] / res[col]
            new_order.extend([col, f"ratio1/{col.split('=')[-1]}"])
        res = res[["batch=1", *new_order]]
    return res.round(3)

def main():
    """main fn"""
    video = pims.Video(dwd_video_if_needed())
    representations_dict = get_representation_dict()
    batch_sizes = [5, 3, 1]
    start_frame = 1000
    end_frame = start_frame + 200

    vres = []
    tmp_dir = Path(TemporaryDirectory().name)
    for _ in range(len(batch_sizes)):
        representations = build_representations_from_cfg(representations_dict)
        vre = VRE(video, representations)
        reprs_setup = {k: representations_dict[k].get("vre_parameters", {}) for k in representations.keys()}
        vres.append(partial(vre, start_frame=start_frame, end_frame=end_frame, export_npy=True,
                            export_png=True, reprs_setup=reprs_setup))

    results = []
    for i, vre in enumerate(vres):
        raw_result = vre(batch_size=batch_sizes[i], output_dir=tmp_dir / f"batch_size_{batch_sizes[i]}")
        results.append(raw_result)
        (final := _process_all(results, batch_sizes[0:i + 1])).to_csv(Path(__file__).parent / "results.csv")
    (final := _process_all(results, batch_sizes)).to_csv(Path(__file__).parent / "results.csv")
    print(final)

if __name__ == "__main__":
    main()
