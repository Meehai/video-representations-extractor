import shutil
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import yaml
import numpy as np
import torch as tr

from vre_video import VREVideo
from vre import VRE
from vre.representations import build_representations_from_cfg
from vre.utils import fetch_resource
from vre.logger import vre_logger as logger

from vre_repository import get_vre_repository

def test_vre_batched():
    video = VREVideo(fetch_resource("test_video.mp4"))
    device = "cuda" if tr.cuda.is_available() else "cpu"
    all_representations_dict = yaml.safe_load(f"""
representations:
  rgb:
    type: color/rgb
    dependencies: []
    parameters: {{}}

  hsv:
    type: color/hsv
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
    type: soft-segmentation/generalized-boundaries
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
      delta: 1
    learned_parameters:
      device: {device}

  normals_svd(depth_dpt):
    type: normals/depth-svd
    dependencies: [depth_dpt]
    parameters:
      sensor_fov: 75
      sensor_size: [3840, 2160]
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
      disk_data_argmax: False
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
      variant: model_1M
      disk_data_argmax: True
    learned_parameters:
      device: {device}
""")

    all_representations = build_representations_from_cfg(all_representations_dict, get_vre_repository())
    np.random.seed(0)
    chosen = list(np.random.choice(all_representations, size=2, replace=False))
    for item in chosen:
        chosen.extend(item.dependencies)
    chosen = list(set(chosen))
    # representations = {k: v for k, v in all_representations.items() if k in chosen}
    logger.info(f"Kept representations: {chosen}")

    tmp_dir = Path("here1" if __name__ == "__main__" else TemporaryDirectory().name)
    tmp_dir_bs = Path("here2" if __name__ == "__main__" else TemporaryDirectory().name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(tmp_dir_bs, ignore_errors=True)

    start_frame = 1000 if __name__ == "__main__" else np.random.randint(0, len(video) - 2)
    end_frame = start_frame + 2
    batch_size = 2 # BS=2 is enough to test this. In examples/ we have a benchmark that tries more values

    vre = VRE(video, chosen)
    vre.set_compute_params(batch_size=1).set_io_parameters(binary_format="npz", image_format="png", compress=True)
    run_meta = vre.run(tmp_dir_bs, frames=list(range(start_frame, end_frame)), output_dir_exists_mode="raise")
    vre.set_compute_params(batch_size=batch_size)
    run_meta_bs = vre(tmp_dir, frames=list(range(start_frame, end_frame)), output_dir_exists_mode="raise")

    unbatched_paths = [run_meta.disk_location.parents[1] / r / ".repr_metadata.json" for r in run_meta.repr_names]
    batched_paths = [run_meta_bs.disk_location.parents[1] / r / ".repr_metadata.json" for r in run_meta.repr_names]
    unbatched_meta = [json.load(open(pth, "r")) for pth in unbatched_paths]
    batched_meta = [json.load(open(pth, "r")) for pth in batched_paths]

    unbatched = np.array([[v["duration"] for v in unbatched_meta[i]["run_stats"].values() if v is not None]
                          for i in range(len(unbatched_meta))]).mean(1)
    batched = np.array([[v["duration"] for v in batched_meta[i]["run_stats"].values() if v is not None]
                        for i in range(len(batched_meta))]).mean(1)
    total_u, total_b = (end_frame - start_frame) * unbatched.sum(), (end_frame - start_frame) * batched.sum()
    for k, u, b in zip(run_meta.repr_names, unbatched, batched):
        logger.info(f"{k}: {u:.2f} vs {b:.2f} ({u / b:.2f})")
    logger.info(f"Total: {total_u:.2f} vs {total_b:.2f} ({total_u / total_b:.2f})")

    for r_name in vre.repr_names:
        for t in range(start_frame, end_frame):
            a = np.load(tmp_dir / r_name / "npz/" / f"{t}.npz")["arr_0"]
            b = np.load(tmp_dir_bs / r_name / "npz/" / f"{t}.npz")["arr_0"]
            assert np.abs(a - b).mean() < 1e-2, (r_name, t)

if __name__ == "__main__":
    test_vre_batched()
