#!/usr/bin/env python3
"""experiment using batches of frames"""
from pathlib import Path
from tempfile import TemporaryDirectory
import pims
import pandas as pd
import torch as tr
import os

from vre import VRE
from vre.logger import vre_logger as logger
from vre.representations import build_representations_from_cfg
from vre.utils import get_project_root

def get_representation_dict() -> dict:
    """setup all representations we want to use, including one representation per device"""
    device = "cuda" if tr.cuda.is_available() else "cpu"
    all_representations_dict: dict[str, dict] = {
        "rgb": {"type": "default", "name": "rgb", "dependencies": [], "parameters": {}},
        "hsv": {"type": "default", "name": "hsv", "dependencies": [], "parameters": {}},
        "dexined": {"type": "edges", "name": "dexined", "dependencies": [],
                    "parameters": {},
                    "device": device},
        "softseg gb": {"type": "soft-segmentation", "name": "generalized_boundaries", "dependencies": [],
                       "parameters": {"use_median_filtering": True, "adjust_to_rgb": True, "max_channels": 3}},
        "canny": {"type": "edges", "name": "canny", "dependencies": [],
                  "parameters": {"threshold1": 100, "threshold2": 200, "aperture_size": 3, "l2_gradient": True}},
        "depth dpt": {"type": "depth", "name": "dpt", "dependencies": [], "parameters": {},
                      "device": device},
        "normals svd (dpth)": {"type": "normals", "name": "depth-svd", "dependencies": ["depth dpt"],
                               "parameters": {"sensor_fov": 75, "sensor_width": 3840,
                                              "sensor_height": 2160, "window_size": 11}},
        "opticalflow rife": {"type": "optical-flow", "name": "rife", "dependencies": [],
                             "parameters": {"compute_backward_flow": False, "uhd": False},
                             "device": device},
        "semantic safeuav torch": {"type": "semantic-segmentation", "name": "safeuav", "dependencies": [],
                                   "parameters": {"train_height": 240, "train_width": 428, "num_classes": 8,
                                                  "weights_file": "safeuav_semantic_0956_pytorch.ckpt",
                                                  "color_map": [[0, 255, 0], [0, 127, 0], [255, 255, 0],
                                                                [255, 255, 255], [255, 0, 0], [0, 0, 255],
                                                                [0, 255, 255], [127, 127, 63]]},
                                   "device": device},
        "halftone": {"type": "soft-segmentation", "name": "python-halftone", "dependencies": [],
                     "parameters": {"sample": 3, "scale": 1, "percentage": 91, "angles": [0, 15, 30, 45],
                                    "antialias": False, "resolution": [240, 426]}},
        "opticalflow raft": {"type": "optical-flow", "name": "raft", "dependencies": [],
                             "parameters": {"small": False, "iters": 20},
                             "device": device},
        "fastsam (s)": {"type": "soft-segmentation", "name": "fastsam", "dependencies": [],
                        "parameters": {"variant": "fastsam-s", "iou": 0.9, "conf": 0.4},
                        "device": device},
        "fastsam (x)": {"type": "soft-segmentation", "name": "fastsam", "dependencies": [],
                        "parameters": {"variant": "fastsam-s", "iou": 0.9, "conf": 0.4},
                        "device": device},
        "mask2former (r50)": {"type": "semantic-segmentation", "name": "mask2former", "dependencies": [],
                              "batch_size": 2, "device": device,
                              "parameters": {"model_id": "49189528_1", "semantic_argmax_only": False}},
        "mask2former (swin)": {"type": "semantic-segmentation", "name": "mask2former", "dependencies": [],
                               "batch_size": 2, "device": device,
                               "parameters": {"model_id": "47429163_0", "semantic_argmax_only": True}},
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
        n_needed += "device" in v # +1
    if n_needed > tr.cuda.device_count():
        logger.info(f"Using 1 gpu. n_needed={n_needed}, n_available={tr.cuda.device_count()}")
        return all_representations_dict
    logger.info(f"Using {tr.cuda.device_count()} GPUs")
    i = 0
    for k in all_representations_dict.keys():
        if "device" in all_representations_dict[k]:
            all_representations_dict[k] = f"cuda:{i}"
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
    video = pims.Video(get_project_root() / "resources/test_video.mp4")
    representations_dict = get_representation_dict()
    batch_sizes = [5, 3, 1]
    start_frame = 1000
    end_frame = start_frame + 200

    tmp_dir = Path(TemporaryDirectory().name)
    vres = [VRE(video, build_representations_from_cfg(representations_dict)) for i in range(len(batch_sizes))]
    results = []
    for i, vre in enumerate(vres):
        output_dir = tmp_dir / f"batch_size_{batch_sizes[i]}"
        raw_result = vre(output_dir=output_dir, batch_size=batch_sizes[i], start_frame=start_frame, end_frame=end_frame)
        results.append(raw_result)
        (final := _process_all(results, batch_sizes[0:i + 1])).to_csv(Path(__file__).parent / "results.csv")
    (final := _process_all(results, batch_sizes)).to_csv(Path(__file__).parent / "results.csv")
    print(final)

if __name__ == "__main__":
    main()
