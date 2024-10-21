# pylint: disable=all
import torch
import torch.nn.functional as F


def median_filter(x, kernel_size=3):
    nb, nc, ny, nx = x.shape
    x = x.reshape(nb * nc, 1, ny, nx)
    x = F.unfold(x, kernel_size, padding=(kernel_size - 1) // 2)
    y = x.median(dim=1, keepdim=True)[0]
    y = F.fold(y, (ny, nx), 1)
    y = y.view(nb, nc, ny, nx)
    return y


def like_an_image(seg, img, sampling_step=5):
    nb, nc, ny, nx = seg.shape
    ni = img.shape[1]
    img = F.interpolate(img, size=(ny, nx), mode='nearest')
    seg = torch.cat((seg, torch.ones((nb, 1, seg.shape[2], seg.shape[3]), dtype=seg.dtype, device=seg.device)), dim=1)

    ys, xs = torch.meshgrid(torch.arange(0, ny - sampling_step, sampling_step),
                            torch.arange(0, nx - sampling_step, sampling_step), indexing="ij")
    seg_spl = seg[:, :, ys, xs].view(nb, nc + 1, -1).permute(0, 2, 1).unsqueeze(1).expand(-1, ni, -1, -1)
    img_spl = img[:, :, ys, xs].view(nb, ni, -1).unsqueeze(3)

    trf = torch.linalg.lstsq(seg_spl, img_spl)[0]
    seg_img = seg.view(nb, nc + 1, -1).permute(0, 2, 1).unsqueeze(1).expand(-1, ni, -1, -1)
    seg_img = seg_img @ trf
    seg_img = seg_img.squeeze(3).view(nb, ni, ny, nx)

    return seg_img


def rgb2hsv(img, epsilon=1e-10):
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    max_rgb, _ = img.max(1)
    min_rgb, argmin_rgb = img.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h / 360, s, v), dim=1)


def hsv2rgb(img):
    assert(img.shape[1] == 3)

    h, s, v = img[:, 0] * 360, img[:, 1], img[:, 2]
    h_ = (h - torch.floor(h / 360) * 360) / 60
    c = s * v
    x = c * (1 - torch.abs(torch.fmod(h_, 2) - 1))

    zero = torch.zeros_like(c)
    y = torch.stack((
        torch.stack((c, x, zero), dim=1),
        torch.stack((x, c, zero), dim=1),
        torch.stack((zero, c, x), dim=1),
        torch.stack((zero, x, c), dim=1),
        torch.stack((x, zero, c), dim=1),
        torch.stack((c, zero, x), dim=1),
    ), dim=0)

    index = torch.repeat_interleave(torch.floor(h_).unsqueeze(1), 3, dim=1).unsqueeze(0).to(torch.long)
    rgb = (y.gather(dim=0, index=index) + (v - c)).squeeze(0)
    return rgb


def normalize_channels(x, whole_block=False):
    _, nc, ny, nx = x.shape
    if whole_block:
        min_c = x.view(x.shape[0], nc * ny * nx).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_c = x.view(x.shape[0], nc * ny * nx).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    else:
        min_c = x.view(x.shape[0], -1, ny * nx).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        max_c = x.view(x.shape[0], -1, ny * nx).max(dim=2, keepdim=True)[0].unsqueeze(-1)
    return (x - min_c) / (max_c - min_c + torch.finfo(x.dtype).eps)

