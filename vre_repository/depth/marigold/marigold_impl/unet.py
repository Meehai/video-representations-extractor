"""
Marigold's denoising UNet in plain torch, inference only. A port of diffusers' (0.30.3) UNet2DConditionModel limited
to the config Marigold uses: 3 cross-attention down blocks + 1 plain one, a cross-attention mid block, 1 plain up block
+ 3 cross-attention ones, linear-projection transformers with GEGLU feed forward, sinusoidal timestep embedding.
Module names match diffusers, so its `state_dict()` loads here with strict=True and the outputs are the same (SDPA).
ResnetBlock2D, Downsample2D, Upsample2D, Attention and MidBlock are shared with the VAE.
"""
import math
import torch as tr
from torch import nn
from torch.nn import functional as F

LAYERS_PER_BLOCK = 2
NORM_EPS = 1e-5

class ResnetBlock2D(nn.Module):
    """norm -> silu -> conv (+ time embedding) -> norm -> silu -> conv, plus a (1x1 conv if needed) skip"""
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int | None, groups: int, eps: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = None if temb_channels is None else nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_shortcut = None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: tr.Tensor, temb: tr.Tensor | None) -> tr.Tensor:
        """(B, C_in, H, W), (B, T) or None -> (B, C_out, H, W)"""
        assert (temb is None) == (self.time_emb_proj is None), (temb is None, self.time_emb_proj is None)
        h = self.conv1(F.silu(self.norm1(x)))
        if temb is not None:
            h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        skip = x if self.conv_shortcut is None else self.conv_shortcut(x)
        return skip + h

class Downsample2D(nn.Module):
    """3x3 conv with stride 2. padding=0 pads right/bottom by 1 instead (the VAE)"""
    def __init__(self, channels: int, padding: int):
        super().__init__()
        assert padding in (0, 1), padding
        self.padding = padding
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=padding)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """(B, C, H, W) -> (B, C, ~H/2, ~W/2)"""
        if self.padding == 0:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)

class Upsample2D(nn.Module):
    """nearest x2 (or to `size`) then 3x3 conv"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: tr.Tensor, size: tuple[int, int] | None = None) -> tr.Tensor:
        """(B, C, H, W) -> (B, C, 2H, 2W) or (B, C, *size)"""
        if size is None:
            return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        return self.conv(F.interpolate(x, size=size, mode="nearest"))

class Attention(nn.Module):
    """multi-head (cross) attention on (B, N, D) tokens. diffusers' Attention + AttnProcessor2_0"""
    def __init__(self, query_dim: int, heads: int, dim_head: int, cross_attention_dim: int | None = None,
                 bias: bool = False):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim or query_dim, heads * dim_head, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim or query_dim, heads * dim_head, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, query_dim)]) # to_out.1 is diffusers' no-op dropout

    def forward(self, x: tr.Tensor, context: tr.Tensor | None = None) -> tr.Tensor:
        """(B, N, D) attends to itself or to context (B, M, D_ctx) -> (B, N, D)"""
        b, context = x.shape[0], x if context is None else context
        q, k, v = [y.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
                   for y in (self.to_q(x), self.to_k(context), self.to_v(context))]
        y = F.scaled_dot_product_attention(q, k, v) # pylint: disable=not-callable
        return self.to_out[0](y.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head))

class GEGLU(nn.Module):
    """linear to 2x, then value * gelu(gate)"""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """(B, N, dim_in) -> (B, N, dim_out)"""
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate) # pylint: disable=not-callable

class FeedForward(nn.Module):
    """GEGLU (4x) -> linear"""
    def __init__(self, dim: int):
        super().__init__()
        # net.1 is diffusers' (no-op) dropout: keeps the linear at net.2 for the state_dict keys
        self.net = nn.ModuleList([GEGLU(dim, dim * 4), nn.Identity(), nn.Linear(dim * 4, dim)])

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """(B, N, D) -> (B, N, D)"""
        for module in self.net:
            x = module(x)
        return x

class BasicTransformerBlock(nn.Module):
    """pre-norm self attention -> cross attention on the context -> feed forward, each with a residual"""
    def __init__(self, dim: int, heads: int, dim_head: int, cross_attention_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn1 = Attention(dim, heads, dim_head)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.attn2 = Attention(dim, heads, dim_head, cross_attention_dim)
        self.norm3 = nn.LayerNorm(dim, eps=1e-5)
        self.ff = FeedForward(dim)

    def forward(self, x: tr.Tensor, context: tr.Tensor) -> tr.Tensor:
        """(B, N, D), (B, M, D_ctx) -> (B, N, D)"""
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        return self.ff(self.norm3(x)) + x

class Transformer2DModel(nn.Module):
    """group norm -> linear in -> 1 transformer block on the H*W tokens -> linear out, plus a residual"""
    def __init__(self, in_channels: int, heads: int, groups: int, cross_attention_dim: int):
        super().__init__()
        dim_head = in_channels // heads
        self.norm = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, heads * dim_head)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(heads * dim_head, heads, dim_head,
                                                                       cross_attention_dim)])
        self.proj_out = nn.Linear(heads * dim_head, in_channels)

    def forward(self, x: tr.Tensor, context: tr.Tensor) -> tr.Tensor:
        """(B, C, H, W), (B, M, D_ctx) -> (B, C, H, W)"""
        b, c, h, w = x.shape
        y = self.proj_in(self.norm(x).permute(0, 2, 3, 1).reshape(b, h * w, c))
        for block in self.transformer_blocks:
            y = block(y, context)
        y = self.proj_out(y).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return y + x

class DownBlock(nn.Module):
    """
    LAYERS_PER_BLOCK x (resnet [-> transformer if heads is not None]) [-> downsample]. Returns the output and the
    skips for the up path. diffusers' CrossAttnDownBlock2D and DownBlock2D
    """
    def __init__(self, in_channels: int, out_channels: int, *, temb_channels: int, groups: int, add_downsample: bool,
                 heads: int | None, cross_attention_dim: int):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels,
                                                    temb_channels, groups, NORM_EPS) for i in range(LAYERS_PER_BLOCK)])
        self.attentions = nn.ModuleList([] if heads is None else [
            Transformer2DModel(out_channels, heads, groups, cross_attention_dim) for _ in range(LAYERS_PER_BLOCK)])
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, padding=1)] if add_downsample else [])

    def forward(self, x: tr.Tensor, temb: tr.Tensor, context: tr.Tensor) -> tuple[tr.Tensor, tuple[tr.Tensor, ...]]:
        """(B, C_in, H, W) -> (B, C_out, H', W'), skips"""
        skips = ()
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            if len(self.attentions) > 0:
                x = self.attentions[i](x, context)
            skips = skips + (x, )
        for downsampler in self.downsamplers:
            x = downsampler(x)
            skips = skips + (x, )
        return x, skips

class MidBlock(nn.Module):
    """resnet -> attention -> resnet. diffusers' UNetMidBlock2DCrossAttn (UNet) and UNetMidBlock2D (VAE)"""
    def __init__(self, channels: int, temb_channels: int | None, groups: int, eps: float, attention: nn.Module):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(channels, channels, temb_channels, groups, eps) for _ in range(2)])
        self.attentions = nn.ModuleList([attention])

    def forward(self, x: tr.Tensor, temb: tr.Tensor | None = None, context: tr.Tensor | None = None) -> tr.Tensor:
        """(B, C, H, W) -> (B, C, H, W)"""
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        return self.resnets[1](x, temb)

class UpBlock(nn.Module):
    """
    LAYERS_PER_BLOCK + 1 x (concat skip -> resnet [-> transformer if heads is not None]) [-> upsample].
    diffusers' CrossAttnUpBlock2D and UpBlock2D
    """
    def __init__(self, in_channels: int, prev_output_channel: int, out_channels: int, *, temb_channels: int,
                 groups: int, add_upsample: bool, heads: int | None, cross_attention_dim: int):
        super().__init__()
        n = LAYERS_PER_BLOCK + 1
        x_channels = [prev_output_channel] + [out_channels] * (n - 1)
        skip_channels = [out_channels] * (n - 1) + [in_channels]
        self.resnets = nn.ModuleList([ResnetBlock2D(x_channels[i] + skip_channels[i], out_channels, temb_channels,
                                                    groups, NORM_EPS) for i in range(n)])
        self.attentions = nn.ModuleList([] if heads is None else [
            Transformer2DModel(out_channels, heads, groups, cross_attention_dim) for _ in range(n)])
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)] if add_upsample else [])

    def forward(self, x: tr.Tensor, skips: tuple[tr.Tensor, ...], temb: tr.Tensor, context: tr.Tensor,
                upsample_size: tuple[int, int] | None) -> tr.Tensor:
        """(B, C_prev, H, W), skips (last one first) -> (B, C_out, H', W')"""
        assert len(skips) == len(self.resnets), (len(skips), len(self.resnets))
        for i, resnet in enumerate(self.resnets):
            x = resnet(tr.cat([x, skips[-1 - i]], dim=1), temb)
            if len(self.attentions) > 0:
                x = self.attentions[i](x, context)
        for upsampler in self.upsamplers:
            x = upsampler(x, upsample_size)
        return x

class TimestepEmbedding(nn.Module):
    """linear -> silu -> linear"""
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """(B, C) -> (B, T)"""
        return self.linear_2(F.silu(self.linear_1(x)))

def timestep_proj(timesteps: tr.Tensor, dim: int) -> tr.Tensor:
    """(B, ) timesteps -> (B, dim) sinusoidal embedding, [cos | sin] (flip_sin_to_cos=True, freq_shift=0)"""
    assert timesteps.ndim == 1 and dim % 2 == 0, (timesteps.shape, dim)
    half_dim = dim // 2
    exponent = -math.log(10000) * tr.arange(start=0, end=half_dim, dtype=tr.float32, device=timesteps.device)
    emb = timesteps[:, None].float() * tr.exp(exponent / half_dim)[None, :]
    return tr.cat([tr.cos(emb), tr.sin(emb)], dim=-1)

class UNet2DConditionModel(nn.Module):
    """forward: (B, C_in, H, W) noisy latent, () timestep, (B, M, D_ctx) context -> (B, C_out, H, W)"""
    def __init__(self, in_channels: int, out_channels: int, block_out_channels: list[int], heads: list[int], *,
                 norm_num_groups: int, cross_attention_dim: int):
        super().__init__()
        n = len(block_out_channels)
        assert n == len(heads) and n >= 2, (block_out_channels, heads)
        ch, temb = block_out_channels, block_out_channels[0] * 4
        kw = {"temb_channels": temb, "groups": norm_num_groups, "cross_attention_dim": cross_attention_dim}
        self.num_upsamplers = n - 1
        self.conv_in = nn.Conv2d(in_channels, ch[0], kernel_size=3, padding=1)
        self.time_embedding = TimestepEmbedding(ch[0], temb)
        # the last down block and the first up block have no attention
        self.down_blocks = nn.ModuleList([DownBlock(ch[max(i - 1, 0)], ch[i], add_downsample=i < n - 1,
                                                    heads=heads[i] if i < n - 1 else None, **kw) for i in range(n)])
        self.mid_block = MidBlock(ch[-1], temb, norm_num_groups, NORM_EPS,
                                  attention=Transformer2DModel(ch[-1], heads[-1], norm_num_groups, cross_attention_dim))
        rch, rheads = ch[::-1], heads[::-1]
        self.up_blocks = nn.ModuleList([UpBlock(rch[min(i + 1, n - 1)], rch[max(i - 1, 0)], rch[i],
                                                add_upsample=i < n - 1, heads=rheads[i] if i > 0 else None, **kw)
                                        for i in range(n)])
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, ch[0], eps=NORM_EPS)
        self.conv_out = nn.Conv2d(ch[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x: tr.Tensor, timestep: tr.Tensor, context: tr.Tensor) -> tr.Tensor:
        """forward"""
        assert x.ndim == 4 and isinstance(timestep, tr.Tensor) and timestep.ndim == 0, (x.shape, timestep)
        # latents not divisible by 2**num_upsamplers: upsample to the skip's size, not x2
        upsample_to_skip = any(d % 2 ** self.num_upsamplers != 0 for d in x.shape[-2:])
        t_emb = timestep_proj(timestep[None].to(x.device).expand(x.shape[0]), self.conv_in.out_channels)
        emb = self.time_embedding(t_emb.to(dtype=x.dtype))

        x = self.conv_in(x)
        skips = (x, )
        for down_block in self.down_blocks:
            x, block_skips = down_block(x, emb, context)
            skips = skips + block_skips
        x = self.mid_block(x, emb, context)
        for i, up_block in enumerate(self.up_blocks):
            n_skips = len(up_block.resnets)
            block_skips, skips = skips[-n_skips:], skips[:-n_skips]
            upsample_size = skips[-1].shape[2:] if upsample_to_skip and i < len(self.up_blocks) - 1 else None
            x = up_block(x, block_skips, emb, context, upsample_size)
        return self.conv_out(F.silu(self.conv_norm_out(x)))
