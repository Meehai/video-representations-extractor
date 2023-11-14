# pylint: disable=all
import torch
import torch.nn.functional as F
from .util import rgb2hsv, normalize_channels, median_filter, like_an_image


def soft_seg(img, max_channels=3, use_median_filtering=False, as_image=False, internal_size=300):
    nb, _, ny, nx = img.shape
    img = normalize_channels(img)

    # rescale
    scale = internal_size / max(ny, nx)
    image_resized = F.interpolate(img, scale_factor=scale, mode='nearest', recompute_scale_factor=False)
    ry, rx = image_resized.shape[2:4]

    # pca data
    img_hsv = rgb2hsv(image_resized)
    map_sim = get_pca_data(img_hsv)

    # pca over patch maps
    map_sim = map_sim.view(nb, -1, ry * rx)
    mean_maps = map_sim.mean(dim=2, keepdim=True)
    map_sim -= mean_maps

    # eig
    val, eig = torch.linalg.eig(map_sim @ map_sim.transpose(2, 1))
    val, eig = val.real, eig.real   # matrix is symmetric

    # project maps on compressed space
    eig_valid = eig[:, :, :max_channels]
    seg = map_sim.transpose(2, 1) @ eig_valid
    seg = seg.view(nb, ry, rx, -1).permute(0, 3, 1, 2)

    # median filtering
    if use_median_filtering:
        seg = median_filter(seg, kernel_size=3)

    # chromatic adjustment to input image colors
    if as_image:
        seg = like_an_image(seg, image_resized, sampling_step=3)

    # resize output segmentations
    seg = F.interpolate(seg, size=(ny, nx), mode='bicubic', align_corners=False)
    seg = normalize_channels(seg, whole_block=True)

    return seg


def get_pca_data(img_hsv, dim_h=15, dim_s=11, dim_v=7, window=16, sample_step=30):
    nb, nc, ny, nx = img_hsv.shape
    dw = window // 2
    img_hsv = normalize_channels(img_hsv)

    # color mix
    h = (img_hsv[:, 0] * (dim_h - 1) + 1).round()
    s = (img_hsv[:, 1] * (dim_s - 1) + 1).round()
    v = (img_hsv[:, 2] * (dim_v - 1) + 1).round()
    img_col = h + (s - 1) * dim_h + (v - 1) * dim_h * dim_s - 1
    img_col = img_col.round().type(torch.long)

    # get reference patch locations
    xs, ys = torch.meshgrid(torch.arange(dw, nx - dw, dw), torch.arange(dw, ny - dw, dw), indexing="ij")
    ys, xs = ys.flatten(), xs.flatten()
    patch_colors = torch.zeros(nb, ys.numel(), dim_h * dim_v * dim_s, dtype=torch.bool, device=img_hsv.device)

    # patch has colors
    for b in range(nb):
        for i in range(ys.numel()):
            idx = img_col[b, ys[i] - dw:ys[i] + dw, xs[i] - dw:xs[i] + dw]
            patch_colors[b, i, idx.flatten()] = True

    # pca prep
    patch_colors = patch_colors.type(torch.float)
    mean_colors = patch_colors.mean(dim=1, keepdim=True)
    patch_colors -= mean_colors

    # eig
    val, eig = torch.linalg.eig(patch_colors.transpose(2, 1) @ patch_colors)
    val, eig = val.real, eig.real   # matrix is symmetric

    # get sampling patch locations
    xs, ys = torch.meshgrid(torch.arange(dw, nx - dw, sample_step), torch.arange(dw, ny - dw, sample_step), indexing="ij")
    ys, xs = ys.flatten(), xs.flatten()

    energy_sum = val.cumsum(dim=1)
    data = torch.zeros(nb, ys.numel(), ny, nx, device=img_hsv.device)
    for b in range(nb):
        # add up smallest values until 50% of total is reached
        max_vector = (energy_sum[b] > energy_sum[b, -1] * 0.5).nonzero()[0]
        eig_valid = eig[b, :, :max_vector + 1]

        for i in range(ys.numel()):
            # sample patch colors
            aux = img_col[b, ys[i] - dw:ys[i] + dw, xs[i] - dw:xs[i] + dw]
            aux_colors = torch.zeros(1, dim_h * dim_v * dim_s, device=img_hsv.device)
            aux_colors[0, aux.flatten()] = 1.
            aux_colors -= mean_colors[b]

            # how much of each color eigenvector is in this patch
            aux_match = (aux_colors @ eig_valid) @ eig_valid.T

            # vector mapped color values for this patch
            c0 = mean_colors[b] - aux_match
            c1 = mean_colors[b] + aux_match
            c0, c1 = c0.clamp(min=0), c1.clamp(min=0)

            # how much do colors in this patch appear across the image
            data[b, i] = c1[0, img_col[b]] / (c0[0, img_col[b]] + c1[0, img_col[b]] + torch.finfo(c0.dtype).eps)

    return data

