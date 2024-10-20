import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import yaml
import pims
import pandas as pd
import numpy as np
import torch as tr

from vre import VRE
from vre.representations import build_representations_from_cfg
from vre.utils import fetch_resource

def sample_representations(all_representations_dict: dict[str, Any], n: int) -> dict:
    np.random.seed(41)
    def _get_deps(all_representations_dict: dict[str, Any], key: str) -> set[str]:
        res = set()
        left = [key]
        while len(left) > 0:
            curr = left.pop()
            res.add(curr)
            left.extend(all_representations_dict[curr]["dependencies"])
        return res

    chosen_ones = np.random.choice(list(all_representations_dict.keys()), size=2, replace=False).tolist()
    res_dict = {}
    for chosen_one in chosen_ones:
        res_dict[chosen_one] = all_representations_dict[chosen_one]
        for dep in _get_deps(all_representations_dict, chosen_one):
            res_dict[dep] = all_representations_dict[dep]
        if len(res_dict) >= n:
            break
    return res_dict

def test_vre_batched():
    video = pims.Video(fetch_resource("test_video.mp4"))
    device = "cuda" if tr.cuda.is_available() else "cpu"
    all_representations_dict = yaml.safe_load(f"""
  rgb:
    type: default/rgb
    dependencies: []
    parameters: {{}}

  hsv:
    type: default/hsv
    dependencies: []
    parameters: {{}}

  halftone:
    type: soft-segmentation/python-halftone
    dependencies: []
    parameters:
      sample: 3
      scale: 1
      percentage: 91
      angles: [0, 15, 30, 45]
      antialias: False
      resolution: [240, 426]

  edges_canny:
    type: edges/canny
    dependencies: []
    parameters:
      threshold1: 100
      threshold2: 200
      aperture_size: 3
      l2_gradient: True

  softseg_gb:
    type: soft-segmentation/generalized_boundaries
    dependencies: []
    parameters:
      use_median_filtering: True
      adjust_to_rgb: True
      max_channels: 3

  edges_dexined:
    type: edges/dexined
    dependencies: []
    parameters: {{}}
    device: {device}

  opticalflow_rife:
    type: optical-flow/rife
    dependencies: []
    parameters:
      uhd: False
      compute_backward_flow: False
    device: {device}

  normals_svd(depth_dpt):
    type: normals/depth-svd
    dependencies: [depth_dpt]
    parameters:
      sensor_fov: 75
      sensor_width: 3840
      sensor_height: 2160
      window_size: 11

  fastsam(s):
    type: soft-segmentation/fastsam
    dependencies: []
    parameters:
      variant: fastsam-s
      iou: 0.9
      conf: 0.4
    device: {device}

  mask2former:
    type: semantic-segmentation/mask2former
    dependencies: []
    parameters:
      model_id: "49189528_1"
      semantic_argmax_only: False
    device: {device}

  # opticalflow raft:
  #   type: optical-flow/raft
  #   dependencies: []
  #   parameters:
  #     inference_height: 720
  #     inference_width: 1280
  #   device: {device}

  depth_dpt:
    type: depth/dpt
    dependencies: []
    parameters: {{}}
    device: {device}

  semantic_safeuav_torch:
    type: semantic-segmentation/safeuav
    dependencies: []
    parameters:
      train_height: 240
      train_width: 428
      num_classes: 8
      color_map: [[0, 255, 0], [0, 127, 0], [255, 255, 0], [255, 255, 255],
                  [255, 0, 0], [0, 0, 255], [0, 255, 255], [127, 127, 63]]
    device: {device}
""")

    # we'll just pick 2 random representations to test here
    representations_dict = sample_representations(all_representations_dict, n=2)
    representations = build_representations_from_cfg(representations_dict)
    representations_bs = build_representations_from_cfg(representations_dict)

    tmp_dir = Path("here1" if __name__ == "__main__" else TemporaryDirectory().name)
    tmp_dir_bs = Path("here2" if __name__ == "__main__" else TemporaryDirectory().name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(tmp_dir_bs, ignore_errors=True)

    start_frame, end_frame = 1000, (1002 if __name__ == "__main__" else 1002)
    batch_size = 2

    vre_bs = VRE(video, representations_bs)
    took_bs = vre_bs(tmp_dir_bs, start_frame=start_frame, end_frame=end_frame, export_npy=True, export_png=True,
                     batch_size=batch_size, output_dir_exists_mode="raise")
    vre = VRE(video, representations)
    took1 = vre(tmp_dir, start_frame=start_frame, end_frame=end_frame, export_npy=True, export_png=True,
                batch_size=1, output_dir_exists_mode="raise")

    both = pd.concat([pd.DataFrame(took1["run_stats"]).mean().rename("unbatched"),
                      pd.DataFrame(took_bs["run_stats"]).mean().rename(f"batch={batch_size}")], axis=1)
    both.loc["total"] = both.sum() * (end_frame - start_frame)
    print(both)

    for representation in vre.representations.keys():
        for t in range(start_frame, end_frame):
            a = np.load(tmp_dir / representation / "npy/" / f"{t}.npz")["arr_0"]
            b = np.load(tmp_dir_bs / representation / "npy/" / f"{t}.npz")["arr_0"]
            assert np.abs(a - b).mean() < 1e-2, (representation, t)

if __name__ == "__main__":
    test_vre_batched()
