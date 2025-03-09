#!/usr/bin/env python3
import os
import torch as tr
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pathlib import Path
from torch.nn import functional as F

from vre.utils import get_project_root, lo, colorize_semantic_segmentation, image_resize_batch
from vre import FFmpegVideo
from vre_repository.optical_flow.raft import FlowRaft
from vre_repository.semantic_segmentation.safeuav import SafeUAV

device = "cpu"#"cuda" if tr.cuda.is_available() else "cpu"

def warp_image(rgb_t: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    rgb_t :: (B, H, W, 3) uint8 [0:255]
    flow :: (B, H, W, 2) float32 [-1:-1]
    """
    image = (tr.tensor(rgb_t).permute(0, 3, 1, 2).float() / 255).to(device)
    flow = tr.tensor(flow).float().to(device)

    H, W = image.shape[-2:]

    # Create normalized meshgrid [-1,1] for grid_sample
    lsw, lsh =  tr.linspace(-1, 1, W, device=image.device), tr.linspace(-1, 1, H, device=image.device),
    grid_x, grid_y = tr.meshgrid(lsw, lsh, indexing="xy")
    grid = tr.stack((grid_x, grid_y), dim=-1)  # (H, W, 2), normalized [-1, 1]

    # Apply flow directly (since it's already in [-1, 1] range)
    new_grid = grid + flow

    # Warp image using grid_sample
    warped = F.grid_sample(image, new_grid, mode="bilinear", align_corners=True)
    warped_numpy = (warped.permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
    return warped_numpy

def vre_inference(model: "Representation", video: "Video", ixs: list[int]) -> np.ndarray:
    model.data = None
    model.compute(video, ixs)
    res = model.data.output
    res_rsz = image_resize_batch(res, video.shape[1], video.shape[2], interpolation="bilinear")
    return res_rsz

def mm(x):
    return (x - x.min()) / (x.max() - x.min())

if __name__ == "__main__":
    video = FFmpegVideo(get_project_root() / "resources/test_video.mp4")
    print(video.shape, video.fps)
    h, w = video.shape[1:3]
    safeuav = SafeUAV(name="safeuav", dependencies=[], disk_data_argmax=True, variant="model_4M")
    raft_r = FlowRaft(name="flow_raft", dependencies=[], inference_width=w, inference_height=h, iters=5,
                        small=False, delta=1)
    raft_l = FlowRaft(name="flow_raft", dependencies=[], inference_width=w, inference_height=h, iters=5,
                        small=False, delta=-1)
    raft_r.device = raft_l.device = safeuav.device = device
    raft_r.vre_setup() if raft_r.setup_called is False else None
    raft_l.vre_setup() if raft_l.setup_called is False else None
    safeuav.vre_setup() if safeuav.setup_called is False else None

    mb = 1
    delta = 30
    raft_r.delta = delta
    raft_l.delta = -delta

    ixs = sorted([np.random.randint(delta, len(video) - delta - 1) for _ in range(mb)])
    ixs = [2154]
    ixs_l = [ix + raft_l.delta for ix in ixs]
    ixs_r = [ix + raft_r.delta for ix in ixs]
    print(f"{ixs=}, {ixs_l=}, {ixs_r=}")

    rgbs = video[ixs]
    rgbs_l = video[ixs_l]
    rgbs_r = video[ixs_r]

    flow_l = vre_inference(raft_l, video, ixs)
    flow_r = vre_inference(raft_r, video, ixs)
    flow_l_img = raft_l.make_images(raft_l.data)
    flow_r_img = raft_r.make_images(raft_r.data)

    print(np.array(flow_l).reshape(-1, 2).mean(0), "\n", np.array(flow_r).reshape(-1, 2).mean(0))
    rgb_warp_l = warp_image(rgbs, flow_l)
    rgb_warp_r = warp_image(rgbs, flow_r)
    mask_l = rgb_warp_l.sum(-1, keepdims=True) != 0
    mask_r = rgb_warp_r.sum(-1, keepdims=True) != 0

    diffs_l = ((rgbs_l.astype(np.float32) - rgb_warp_l).__abs__() * mask_l).sum(-1)
    diffs_r = ((rgbs_r.astype(np.float32) - rgb_warp_r).__abs__() * mask_r).sum(-1)
    diffs_l2 = ((rgbs_l.astype(np.float32) - rgb_warp_r).__abs__() * mask_l).sum(-1)
    diffs_r2 = ((rgbs_r.astype(np.float32) - rgb_warp_l).__abs__() * mask_r).sum(-1)
    print(diffs_l.mean(), diffs_l2.mean())
    print(diffs_r.mean(), diffs_r2.mean())

    breakpoint()

# sema = vre_inference(safeuav, video, ixs).argmax(-1)
# sema_img = safeuav.make_images(safeuav.data)
# sema_l = vre_inference(safeuav, video, ixs_l).argmax(-1)
# sema_l_img = safeuav.make_images(safeuav.data)
# sema_r = vre_inference(safeuav, video, ixs_r).argmax(-1)
# sema_r_img = safeuav.make_images(safeuav.data)

# sema_warp_l = warp_image_torch(sema[..., None], flow_l)[..., 0].round().astype(np.uint8)
# sema_warp_r = warp_image_torch(sema[..., None], flow_r)[..., 0].round().astype(np.uint8)

# sema_warp_l_img = colorize_semantic_segmentation(sema_warp_l, safeuav.classes, safeuav.color_map) * Mask_l
# sema_warp_r_img = colorize_semantic_segmentation(sema_warp_r, safeuav.classes, safeuav.color_map) * Mask_r
# red = np.array([[0, 0, 0], [255, 0, 0]])
# diff_sema_l = ((sema_l != sema_warp_l) * Mask_l[..., 0]).astype(int)
# diff_sema_r = ((sema_r != sema_warp_r) * Mask_r[..., 0]).astype(int)
# score = 1 - (diff_sema_l + diff_sema_r) / 2

# for i in range(mb):
#     fig, ax = plt.subplots(3, 3, figsize=(20, 8))
#     ax[0, 0].set_title(f"T={ixs_l[i]}", fontsize=14, fontweight="bold")
#     ax[0, 0].imshow(rgbs_l[i])
#     ax[0, 1].set_title(f"warp {ixs[i]}->{ixs_l[i]}", fontsize=14, fontweight="bold")
#     ax[0, 1].imshow(rgb_warp_l[i].round().astype(np.uint8))
#     ax[0, 2].set_title(f"Diff: {diffs_l.mean().item():.2f}", fontsize=14, fontweight="bold")
#     ax[0, 2].imshow(mm(diffs_l[i]))
#     # ax[0, 3].set_title(f"Sema T={ixs_l[i]}", fontsize=14, fontweight="bold")
#     # ax[0, 3].imshow(sema_l_img[i])
#     # ax[0, 4].imshow(sema_warp_l_img[i])
#     # ax[0, 5].set_title(f"Diff: {diff_sema_l[i].mean()*100:.2f}%", fontsize=14, fontweight="bold")
#     # ax[0, 5].imshow(red[diff_sema_l[i]])

#     ax[1, 0].set_title(f"T={ixs[i]}", fontsize=14, fontweight="bold")
#     ax[1, 0].imshow(video[ixs[i]])
#     ax[1, 1].set_title(f"flow {ixs[i]}->{ixs_l[i]}", fontsize=14, fontweight="bold")
#     ax[1, 1].imshow(flow_l_img[i])
#     ax[1, 2].set_title(f"flow {ixs[i]}->{ixs_r[i]}", fontsize=14, fontweight="bold")
#     ax[1, 2].imshow(flow_r_img[i])
#     # ax[1, 3].set_title(f"Consistency: {score[i].mean() * 100:.2f}%", fontsize=14, fontweight="bold")
#     # ax[1, 3].imshow(sema_img[i])
#     # ax[1, 4].imshow(video[ixs[i]] * 0)
#     # ax[1, 5].imshow(video[ixs[i]] * 0)

#     ax[2, 0].set_title(f"T={ixs_r[i]}", fontsize=14, fontweight="bold")
#     ax[2, 0].imshow(rgbs_r[i])
#     ax[2, 1].set_title(f"warp {ixs[i]}->{ixs_r[i]}", fontsize=14, fontweight="bold")
#     ax[2, 1].imshow(rgb_warp_r[i].round().astype(np.uint8))
#     ax[2, 2].set_title(f"Diff: {diffs_r.mean().item():.2f}", fontsize=14, fontweight="bold")
#     ax[2, 2].imshow(mm(diffs_r[i]))
#     # ax[2, 3].set_title(f"Sema T={ixs_r[i]}", fontsize=14, fontweight="bold")
#     # ax[2, 3].imshow(sema_r_img[i])
#     # ax[2, 4].imshow(sema_warp_r_img[i])
#     # ax[2, 5].set_title(f"Diff: {diff_sema_r[i].mean() * 100:.2f}%", fontsize=14, fontweight="bold")
#     # ax[2, 5].imshow(red[diff_sema_r[i]])

# fig.tight_layout()
# plt.show()
