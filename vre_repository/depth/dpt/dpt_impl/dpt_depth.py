"""
DPT depth (MiDaS, "dpt_large") in plain torch, inference only: ViT-L/16 backbone, readout "project", non-negative
head. Module names match the checkpoint (`pretrained.*` = backbone + reassemble, `scratch.*` = decoder).
"""
import torch as tr
from torch import nn

from .blocks import FeatureFusionBlock, ProjectReadout
from .vit import VisionTransformer

def _make_reassemble(vit_dim: int, features: int, resample: nn.Module) -> nn.Sequential:
    """readout projection, 2 parameter-free placeholders (tokens -> grid, done in DPTEncoder.forward), 1x1 conv,
    resample. The placeholders keep the checkpoint's indices (act_postprocessN.{0,3,4})."""
    return nn.Sequential(ProjectReadout(vit_dim), nn.Identity(), nn.Identity(),
                         nn.Conv2d(vit_dim, features, kernel_size=1, stride=1, padding=0), resample)

class DPTEncoder(nn.Module):
    """(B, 3, H, W) -> 4 maps: strides 4, 8, 16, 32 with 256, 512, 1024, 1024 channels"""
    def __init__(self):
        super().__init__()
        vit_dim = 1024
        self.model = VisionTransformer(out_blocks=(5, 11, 17, 23), dim=vit_dim)
        self.act_postprocess1 = _make_reassemble(vit_dim, 256, nn.ConvTranspose2d(256, 256, kernel_size=4, stride=4))
        self.act_postprocess2 = _make_reassemble(vit_dim, 512, nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2))
        self.act_postprocess3 = _make_reassemble(vit_dim, 1024, nn.Identity())
        self.act_postprocess4 = _make_reassemble(vit_dim, 1024,
                                                 nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1))

    def forward(self, x: tr.Tensor) -> list[tr.Tensor]:
        """forward"""
        grid = (x.shape[2] // self.model.patch_size, x.shape[3] // self.model.patch_size)
        posts = [self.act_postprocess1, self.act_postprocess2, self.act_postprocess3, self.act_postprocess4]
        return [post[3:](post[0](tokens).transpose(1, 2).unflatten(2, grid))
                for post, tokens in zip(posts, self.model(x), strict=True)]

class DPTDecoder(nn.Module):
    """4 encoder maps -> (B, 1, H, W): 3x3 convs to `features` channels, fusion (coarse to fine), head"""
    def __init__(self, in_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024), features: int = 256):
        super().__init__()
        self.layer1_rn = nn.Conv2d(in_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(in_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(in_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(in_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.refinenet1 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet4 = FeatureFusionBlock(features) # no skip input: its resConfUnit1 is unused, kept for strict load
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, layers: list[tr.Tensor]) -> tr.Tensor:
        """forward"""
        assert len(layers) == 4, len(layers)
        path_4 = self.refinenet4(self.layer4_rn(layers[3]))
        path_3 = self.refinenet3(path_4, self.layer3_rn(layers[2]))
        path_2 = self.refinenet2(path_3, self.layer2_rn(layers[1]))
        path_1 = self.refinenet1(path_2, self.layer1_rn(layers[0]))
        return self.output_conv(path_1)

class DPTDepthModel(nn.Module):
    """(B, 3, H, W) in [-1, 1], H and W multiples of 32 -> (B, H, W) non-negative inverse relative depth"""
    def __init__(self):
        super().__init__()
        self.pretrained = DPTEncoder()
        self.scratch = DPTDecoder()

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        assert x.ndim == 4 and x.shape[1] == 3 and x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, x.shape
        return self.scratch(self.pretrained(x)).squeeze(dim=1)
