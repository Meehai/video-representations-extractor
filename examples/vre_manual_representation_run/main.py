import os
import torch as tr
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from vre import FFmpegVideo
from vre.utils import (
    get_project_root, collage_fn, image_write, image_resize, vre_yaml_load, ReprOut, MemoryData, colorize_depth, lo)
from vre_repository.depth.marigold import Marigold
from vre_repository.normals.depth_svd import DepthNormalsSVD
os.environ["VRE_DEVICE"] = device = "cuda" if tr.cuda.is_available() else "cpu"

if __name__ == "__main__":
    video = FFmpegVideo(get_project_root() / "resources/test_video.mp4")
    print(video.shape, video.fps)

    all_representations_dict = vre_yaml_load(open(Path.cwd() / "cfg.yaml", "r"))

    device = "cuda" if tr.cuda.is_available() else "cpu"
    ixs = [2303, 3392]
    print(ixs)
    # decomment this and depth.data/setup/free below if you want to store the depth and work on normals only for example
    # depth = SimpleNamespace(data=ReprOut(frames=video[ixs], output=MemoryData(np.load("data.npz")["arr_0"]), key=ixs),
    #                         make_images=lambda data: (colorize_depth(data.output,
    #                                                                  percentiles=[1, 95]) * 255).astype(np.uint8))
    depth = Marigold(name="marigold", dependencies=[], variant="marigold-lcm-v1-0", denoising_steps=4, ensemble_size=1,
                     processing_resolution=768)
    depth.device = device
    normals = DepthNormalsSVD(name="normals1", dependencies=[depth], sensor_fov=75,
                             sensor_size=(3840, 2160), window_size=11)

    depth.vre_setup()
    depth.data = None
    np.random.seed(42)
    tr.random.manual_seed(42)
    depth.compute(video, ixs)
    # np.savez_compressed("data.npz", depth.data.output)
    # depth.data = ReprOut(frames=video[ixs], output=MemoryData(np.load("data.npz")["arr_0"]), key=ixs)
    normals.data = None
    normals.compute(video, ixs)
    y_depth_img = depth.make_images(depth.data)
    y_normals_img = normals.make_images(normals.data)
    for i in range(len(ixs)):
        res = [image_resize(depth.data.frames[i], *y_depth_img[i].shape[0:2]), y_depth_img[i], y_normals_img[i]]
        image_write(collage_fn(res, rows_cols=(1, -1)), f"res_{ixs[i]}.png")
    # depth.vre_free()
