"""
ViT-L/16 (384) in plain torch, inference only: the DPT backbone. A port of timm's (1.0.9) VisionTransformer
("vit_large_patch16_384") as DPT drives it: module names match, so the checkpoint's `pretrained.model.*` loads with
strict=True and the outputs are the same (SDPA attention, learned position embeddings resized bilinearly to the input's
patch grid). The forward returns the tokens after the `out_blocks` blocks, which is all DPT reads.
"""
import math
import torch as tr
from torch import nn
from torch.nn import functional as F

class PatchEmbed(nn.Module):
    """(B, 3, H, W) -> (B, H//p * W//p, D)"""
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    """multi-head self attention (SDPA) with a fused qkv projection"""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, (dim, num_heads)
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """(B, N, D) -> (B, N, D)"""
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, d // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v) # pylint: disable=not-callable
        return self.proj(y.transpose(1, 2).reshape(b, n, d))

class Mlp(nn.Module):
    """fc -> gelu (erf) -> fc"""
    def __init__(self, dim: int, mlp_ratio: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        return self.fc2(F.gelu(self.fc1(x))) # pylint: disable=not-callable

class Block(nn.Module):
    """pre-norm transformer block"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, layer_norm_eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class VisionTransformer(nn.Module):
    """
    ViT with a CLS token. forward: (B, 3, H, W), H and W multiples of the patch size -> list of the
    (B, 1 + H//p * W//p, D) tokens after each block in `out_blocks`. Token 0 is CLS, the rest are row-major patches.
    """
    def __init__(self, out_blocks: tuple[int, ...], *, dim: int = 1024, depth: int = 24, num_heads: int = 16,
                 mlp_ratio: int = 4, patch_size: int = 16, image_size: int = 384, layer_norm_eps: float = 1e-6):
        super().__init__()
        assert len(out_blocks) > 0 and all(0 <= i < depth for i in out_blocks), (out_blocks, depth)
        self.out_blocks = out_blocks
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(tr.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(tr.zeros(1, (image_size // patch_size) ** 2 + 1, dim))
        self.patch_embed = PatchEmbed(dim, patch_size)
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio, layer_norm_eps) for _ in range(depth)])
        # the final norm and the ImageNet classifier: unused (DPT reads block outputs), kept for strict load
        self.norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.head = nn.Linear(dim, 1000)

    def resize_pos_embed(self, grid_h: int, grid_w: int) -> tr.Tensor:
        """the (1, 1 + grid_h * grid_w, D) position embeddings, the patch part bilinearly resized to the grid"""
        n_positions = self.pos_embed.shape[1] - 1
        side = int(math.sqrt(n_positions))
        assert side * side == n_positions, n_positions
        patch_pos = self.pos_embed[0, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode="bilinear")
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
        return tr.cat([self.pos_embed[:, :1], patch_pos], dim=1)

    def forward(self, x: tr.Tensor) -> list[tr.Tensor]:
        """forward"""
        b, _, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, (x.shape, self.patch_size)
        tokens = tr.cat([self.cls_token.expand(b, -1, -1), self.patch_embed(x)], dim=1)
        tokens = tokens + self.resize_pos_embed(h // self.patch_size, w // self.patch_size)
        res = []
        for i, block in enumerate(self.blocks[0:max(self.out_blocks) + 1]):
            tokens = block(tokens)
            if i in self.out_blocks:
                res.append(tokens)
        return res
