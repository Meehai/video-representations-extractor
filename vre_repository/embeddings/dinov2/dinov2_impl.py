"""
DINOv2 ViT in plain torch, inference only. A port of transformers' Dinov2Model (>= 4.45): module names match it, so
`Dinov2Model.state_dict()` loads here with strict=True and the outputs are the same (SDPA attention, bicubic position
embedding interpolation to the patch grid).
"""
import torch as tr
from torch import nn
from torch.nn import functional as F

class Dinov2PatchEmbeddings(nn.Module):
    """(B, 3, H, W) -> (B, H//p * W//p, D)"""
    def __init__(self, hidden_size: int, patch_size: int):
        super().__init__()
        self.projection = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        return self.projection(x).flatten(2).transpose(1, 2)

class Dinov2Embeddings(nn.Module):
    """CLS token + patch tokens + position embeddings (bicubic-interpolated to the input's patch grid)"""
    def __init__(self, hidden_size: int, patch_size: int, image_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(tr.zeros(1, 1, hidden_size))
        self.mask_token = nn.Parameter(tr.zeros(1, hidden_size)) # training only (masked patches), kept for strict load
        self.patch_embeddings = Dinov2PatchEmbeddings(hidden_size, patch_size)
        self.position_embeddings = nn.Parameter(tr.zeros(1, (image_size // patch_size) ** 2 + 1, hidden_size))

    def interpolate_pos_encoding(self, n_patches: int, height: int, width: int) -> tr.Tensor:
        """the (1, 1 + n_patches, D) position embeddings for a (height, width) input"""
        n_positions = self.position_embeddings.shape[1] - 1
        if n_patches == n_positions and height == width:
            return self.position_embeddings
        side = int(n_positions ** 0.5)
        assert side * side == n_positions, n_positions
        dim = self.position_embeddings.shape[-1]
        patch_pos = self.position_embeddings[:, 1:].reshape(1, side, side, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos.to(tr.float32), size=(height // self.patch_size, width // self.patch_size),
                                  mode="bicubic", align_corners=False).to(self.position_embeddings.dtype)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return tr.cat([self.position_embeddings[:, :1], patch_pos], dim=1)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """(B, 3, H, W) -> (B, 1 + H//p * W//p, D)"""
        b, _, h, w = x.shape
        patches = self.patch_embeddings(x.to(self.patch_embeddings.projection.weight.dtype))
        tokens = tr.cat([self.cls_token.expand(b, -1, -1), patches], dim=1)
        return tokens + self.interpolate_pos_encoding(patches.shape[1], h, w)

class Dinov2SelfAttention(nn.Module):
    """multi-head self attention (SDPA), without the output projection"""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0, (hidden_size, num_heads)
        self.num_heads = num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """(B, N, D) -> (B, N, D)"""
        b, n, d = x.shape
        q, k, v = [lin(x).view(b, n, self.num_heads, d // self.num_heads).permute(0, 2, 1, 3)
                   for lin in (self.query, self.key, self.value)]
        y = F.scaled_dot_product_attention(q, k, v) # pylint: disable=not-callable
        return y.permute(0, 2, 1, 3).reshape(b, n, d)

class Dinov2SelfOutput(nn.Module):
    """the attention's output projection"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        return self.dense(x)

class Dinov2Attention(nn.Module):
    """self attention + output projection"""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention = Dinov2SelfAttention(hidden_size, num_heads)
        self.output = Dinov2SelfOutput(hidden_size)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        return self.output(self.attention(x))

class Dinov2LayerScale(nn.Module):
    """per-channel learned scale"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.lambda1 = nn.Parameter(tr.ones(hidden_size))

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        return x * self.lambda1

class Dinov2MLP(nn.Module):
    """fc -> gelu (erf) -> fc"""
    def __init__(self, hidden_size: int, mlp_ratio: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * mlp_ratio)
        self.fc2 = nn.Linear(hidden_size * mlp_ratio, hidden_size)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        return self.fc2(F.gelu(self.fc1(x))) # pylint: disable=not-callable

class Dinov2Layer(nn.Module):
    """pre-norm transformer block with layer scale"""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int, layer_norm_eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = Dinov2Attention(hidden_size, num_heads)
        self.layer_scale1 = Dinov2LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = Dinov2MLP(hidden_size, mlp_ratio)
        self.layer_scale2 = Dinov2LayerScale(hidden_size)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))

class Dinov2Encoder(nn.Module):
    """the stack of layers"""
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, mlp_ratio: int, layer_norm_eps: float):
        super().__init__()
        self.layer = nn.ModuleList([Dinov2Layer(hidden_size, num_heads, mlp_ratio, layer_norm_eps)
                                    for _ in range(num_layers)])

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        for layer in self.layer:
            x = layer(x)
        return x

class Dinov2Model(nn.Module):
    """
    DINOv2 ViT. forward: normalized (B, 3, H, W) -> last_hidden_state (B, 1 + H//14 * W//14, D) after the final
    layernorm. Token 0 is CLS (transformers' pooler_output), the rest are the patch tokens in row-major order.
    """
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, mlp_ratio: int = 4, patch_size: int = 14,
                 image_size: int = 518, layer_norm_eps: float = 1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.embeddings = Dinov2Embeddings(hidden_size, patch_size, image_size)
        self.encoder = Dinov2Encoder(num_layers, hidden_size, num_heads, mlp_ratio, layer_norm_eps)
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        assert x.ndim == 4 and x.shape[1] == 3, x.shape
        return self.layernorm(self.encoder(self.embeddings(x)))
