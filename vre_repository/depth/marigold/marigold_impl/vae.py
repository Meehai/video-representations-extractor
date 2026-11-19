"""
Marigold's VAE in plain torch, inference only. A port of diffusers' (0.30.3) AutoencoderKL: encoder + quant_conv
(image -> latent moments) and post_quant_conv + decoder (latent -> image). Module names match diffusers, so its
`state_dict()` loads here with strict=True and the outputs are the same.
"""
import torch as tr
from torch import nn
from torch.nn import functional as F

from .unet import Attention, Downsample2D, MidBlock, ResnetBlock2D, Upsample2D

EPS = 1e-6
LAYERS_PER_BLOCK = 2
LATENT_CHANNELS = 4

class VAEAttention(Attention):
    """single-head self attention over the H*W pixels, with a group norm on the input and a residual"""
    def __init__(self, channels: int, groups: int):
        super().__init__(channels, heads=1, dim_head=channels, bias=True)
        self.group_norm = nn.GroupNorm(groups, channels, eps=EPS)

    def forward(self, x: tr.Tensor, context: tr.Tensor | None = None) -> tr.Tensor:
        """(B, C, H, W) -> (B, C, H, W)"""
        assert context is None, context.shape
        b, c, h, w = x.shape
        y = super().forward(self.group_norm(x.view(b, c, h * w)).transpose(1, 2))
        return y.transpose(1, 2).reshape(b, c, h, w) + x

class DownEncoderBlock2D(nn.Module):
    """LAYERS_PER_BLOCK x resnet [-> downsample]"""
    def __init__(self, in_channels: int, out_channels: int, groups: int, add_downsample: bool):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels,
                                                    None, groups, EPS) for i in range(LAYERS_PER_BLOCK)])
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, padding=0)] if add_downsample else [])

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """(B, C_in, H, W) -> (B, C_out, H', W')"""
        for resnet in self.resnets:
            x = resnet(x, None)
        for downsampler in self.downsamplers:
            x = downsampler(x)
        return x

class UpDecoderBlock2D(nn.Module):
    """LAYERS_PER_BLOCK + 1 x resnet [-> upsample]"""
    def __init__(self, in_channels: int, out_channels: int, groups: int, add_upsample: bool):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels,
                                                    None, groups, EPS) for i in range(LAYERS_PER_BLOCK + 1)])
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)] if add_upsample else [])

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """(B, C_in, H, W) -> (B, C_out, H', W')"""
        for resnet in self.resnets:
            x = resnet(x, None)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x

def _mid_block(channels: int, groups: int) -> MidBlock:
    """resnet -> VAEAttention -> resnet"""
    return MidBlock(channels, temb_channels=None, groups=groups, eps=EPS, attention=VAEAttention(channels, groups))

class Encoder(nn.Module):
    """(B, 3, H, W) image -> (B, 2 * LATENT_CHANNELS, H/8, W/8) latent moments (mean, logvar)"""
    def __init__(self, block_out_channels: list[int], groups: int):
        super().__init__()
        ch, n = block_out_channels, len(block_out_channels)
        self.conv_in = nn.Conv2d(3, ch[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([DownEncoderBlock2D(ch[max(i - 1, 0)], ch[i], groups,
                                                             add_downsample=i < n - 1) for i in range(n)])
        self.mid_block = _mid_block(ch[-1], groups)
        self.conv_norm_out = nn.GroupNorm(groups, ch[-1], eps=EPS)
        self.conv_out = nn.Conv2d(ch[-1], 2 * LATENT_CHANNELS, kernel_size=3, padding=1)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        x = self.conv_in(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))

class Decoder(nn.Module):
    """(B, LATENT_CHANNELS, h, w) latent -> (B, 3, 8h, 8w) image"""
    def __init__(self, block_out_channels: list[int], groups: int):
        super().__init__()
        rch, n = block_out_channels[::-1], len(block_out_channels)
        self.conv_in = nn.Conv2d(LATENT_CHANNELS, rch[0], kernel_size=3, padding=1)
        self.mid_block = _mid_block(rch[0], groups)
        self.up_blocks = nn.ModuleList([UpDecoderBlock2D(rch[max(i - 1, 0)], rch[i], groups, add_upsample=i < n - 1)
                                        for i in range(n)])
        self.conv_norm_out = nn.GroupNorm(groups, rch[-1], eps=EPS)
        self.conv_out = nn.Conv2d(rch[-1], 3, kernel_size=3, padding=1)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        # back to the weights' dtype after the mid block (only matters under autocast), as diffusers does
        x = self.mid_block(self.conv_in(x)).to(self.conv_out.weight.dtype)
        for up_block in self.up_blocks:
            x = up_block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))

class AutoencoderKL(nn.Module):
    """the KL VAE parts Marigold calls: encoder -> quant_conv and post_quant_conv -> decoder (no forward)"""
    def __init__(self, block_out_channels: list[int], norm_num_groups: int):
        super().__init__()
        self.encoder = Encoder(block_out_channels, norm_num_groups)
        self.decoder = Decoder(block_out_channels, norm_num_groups)
        self.quant_conv = nn.Conv2d(2 * LATENT_CHANNELS, 2 * LATENT_CHANNELS, 1)
        self.post_quant_conv = nn.Conv2d(LATENT_CHANNELS, LATENT_CHANNELS, 1)
