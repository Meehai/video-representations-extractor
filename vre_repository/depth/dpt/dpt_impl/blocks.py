"""DPT building blocks: the readout projection (encoder side) and the fusion blocks (decoder side)"""
import torch as tr
from torch import nn
from torch.nn import functional as F

class ProjectReadout(nn.Module):
    """(B, 1 + N, D) tokens -> (B, N, D): each patch token concatenated with the CLS token, projected back to D"""
    def __init__(self, dim: int):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, 1:])
        return self.project(tr.cat((x[:, 1:], readout), -1))

class ResidualConvUnit(nn.Module):
    """x + conv(relu(conv(relu(x))))"""
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """forward"""
        return self.conv2(F.relu(self.conv1(F.relu(x)))) + x

class FeatureFusionBlock(nn.Module):
    """(x [+ residual(skip)]) -> residual -> 2x bilinear upsample (align_corners) -> 1x1 conv"""
    def __init__(self, features: int):
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
        self.resConfUnit1 = ResidualConvUnit(features) # pylint: disable=invalid-name
        self.resConfUnit2 = ResidualConvUnit(features) # pylint: disable=invalid-name

    def forward(self, x: tr.Tensor, skip: tr.Tensor | None = None) -> tr.Tensor:
        """forward"""
        if skip is not None:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(x)
