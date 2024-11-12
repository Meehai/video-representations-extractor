import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import yaml
import pandas as pd
import numpy as np
import torch as tr

from vre import VRE
from vre.representations import build_representations_from_cfg
from vre.utils import fetch_resource, FFmpegVideo
from vre.logger import vre_logger as logger

def test_vre_batched():
    video = FFmpegVideo(fetch_resource("test_video.mp4"))
    device = "cuda" if tr.cuda.is_available() else "cpu"
    all_representations_dict = yaml.safe_load(f"""
representations:
  rgb:
    type: default/rgb
    dependencies: []
    parameters: {{}}

  hsv:
    type: default/hsv
    dependencies: [rgb]
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
    learned_parameters:
      device: {device}

  opticalflow_rife:
    type: optical-flow/rife
    dependencies: []
    parameters:
      uhd: False
      compute_backward_flow: False
    learned_parameters:
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
    learned_parameters:
      device: {device}

  mask2former:
    type: semantic-segmentation/mask2former
    dependencies: []
    parameters:
      model_id: "49189528_1"
      semantic_argmax_only: False
    learned_parameters:
      device: {device}

  # opticalflow raft:
  #   type: optical-flow/raft
  #   dependencies: []
  #   parameters:
  #     inference_height: 720
  #     inference_width: 1280
  #   learned_parameters:
  #     device: {device}

  depth_dpt:
    type: depth/dpt
    dependencies: []
    parameters: {{}}
    learned_parameters:
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
      semantic_argmax_only: True
    learned_parameters:
      device: {device}
""")

    all_representations = build_representations_from_cfg(all_representations_dict)
    np.random.seed(0)
    chosen = np.random.choice(list(all_representations.keys()), size=2, replace=False)
    representations = {k: v for k, v in all_representations.items() if k in chosen}
    logger.info(f"Kept representations: {representations}")

    tmp_dir = Path("here1" if __name__ == "__main__" else TemporaryDirectory().name)
    tmp_dir_bs = Path("here2" if __name__ == "__main__" else TemporaryDirectory().name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(tmp_dir_bs, ignore_errors=True)

    start_frame = 1000 if __name__ == "__main__" else np.random.randint(0, len(video) - 2)
    end_frame = start_frame + 2
    batch_size = 2 # BS=2 is enough to test this. In examples/ we have a becnhmark that tries more values

    vre = VRE(video, representations)
    vre.set_compute_params(batch_size=1).set_io_parameters(binary_format="npz", image_format="png", compress=True)
    took1 = vre(tmp_dir_bs, frames=list(range(start_frame, end_frame)), output_dir_exists_mode="raise")
    vre.set_compute_params(batch_size=batch_size)
    took_bs = vre(tmp_dir, frames=list(range(start_frame, end_frame)), output_dir_exists_mode="raise")

    both = pd.concat([pd.DataFrame(took1["run_stats"]).mean().rename("unbatched"),
                      pd.DataFrame(took_bs["run_stats"]).mean().rename(f"batch={batch_size}")], axis=1)
    both.loc["total"] = both.sum() * (end_frame - start_frame)
    print(both)

    for representation in vre.representations.keys():
        for t in range(start_frame, end_frame):
            a = np.load(tmp_dir / representation / "npz/" / f"{t}.npz")["arr_0"]
            b = np.load(tmp_dir_bs / representation / "npz/" / f"{t}.npz")["arr_0"]
            assert np.abs(a - b).mean() < 1e-2, (representation, t)

if __name__ == "__main__":
    test_vre_batched()
