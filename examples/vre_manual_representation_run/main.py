import os
import torch as tr
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pathlib import Path

from vre.representations import build_representations_from_cfg
from vre.utils import get_project_root, FFmpegVideo, collage_fn, image_write, image_resize
from vre_repository import get_vre_repository
os.environ["VRE_DEVICE"] = device = "cuda" if tr.cuda.is_available() else "cpu"

if __name__ == "__main__":
    video = FFmpegVideo(get_project_root() / "resources/test_video.mp4")
    print(video.shape, video.fps)

    all_representations_dict = OmegaConf.load(open(Path.cwd() / "cfg.yaml", "r"))

    device = "cuda" if tr.cuda.is_available() else "cpu"

    representations = build_representations_from_cfg(all_representations_dict, representation_types=get_vre_repository())
    name_to_repr = {r.name: r for r in representations}
    print(representations)

    # inference setup (this is done inside VRE's main loop at run() as well)
    depth, normals = name_to_repr["depth_marigold"], name_to_repr["normals_svd(depth_marigold)"]
    depth.vre_setup() if depth.setup_called is False else None

    np.random.seed(43)
    mb = 2
    ixs = sorted([np.random.randint(0, len(video) - 1) for _ in range(mb)])
    print(ixs)

    depth.data = normals.data = None
    depth.compute(video, ixs)
    normals.compute(video, ixs)
    y_depth_img = depth.make_images(depth.data)
    y_normals_img = normals.make_images(normals.data)
    for i in range(mb):
        res = [image_resize(depth.data.frames[i], *y_depth_img[i].shape[0:2]), y_depth_img[i], y_normals_img[i]]
        image_write(collage_fn(res, rows_cols=(1, 3)), f"res_{ixs[i]}.png")
    depth.vre_free()

    breakpoint()
