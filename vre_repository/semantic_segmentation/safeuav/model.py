"""SafeUAV Model implementation"""
# pylint: disable=all
from __future__ import annotations
import torch as tr
import torch.nn as nn
import torch.nn.functional as F

def conv(d_in: int, d_out: int, kernel_size: int | tuple[int, int], padding: int,
         stride: int, dilation: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=d_in, out_channels=d_out, kernel_size=kernel_size, padding=padding,
                  stride=stride, dilation=dilation),
        nn.BatchNorm2d(num_features=d_out),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )

def conv_t(d_in: int, d_out: int, kernel_size: int | tuple[int, int], padding: int,
            stride: int, dilation: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=d_in, out_channels=d_out, kernel_size=kernel_size, padding=padding,
                           stride=stride, dilation=dilation),
        nn.BatchNorm2d(num_features=d_out),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )

def cat_fn(a: tr.Tensor, b: tr.Tensor) -> tr.Tensor:
    diff_up, diff_left = b.shape[-2] - a.shape[-2], b.shape[-1] - a.shape[-1]
    a = F.pad(a, (0, diff_left, 0, diff_up))
    c = tr.cat([a, b], dim=1)
    return c

class EncoderMap2Map(nn.Module):
    def __init__(self, d_in: int, n_filters: int, dropout: float=0):
        super().__init__()
        self.d_in = d_in
        self.n_filters = n_filters
        self.dropout = dropout

        self.conv1 = conv(d_in=d_in, d_out=n_filters, kernel_size=3, padding=1,
                          stride=1, dilation=1, dropout=dropout)
        self.conv2 = conv(d_in=n_filters, d_out=n_filters, kernel_size=3, padding=1,
                          stride=1, dilation=1, dropout=dropout)
        self.conv3 = conv(d_in=n_filters, d_out=n_filters, kernel_size=3, padding=1,
                          stride=2, dilation=1, dropout=dropout)
        self.conv4 = conv(d_in=n_filters, d_out=n_filters * 2, kernel_size=3, padding=1,
                          stride=1, dilation=1, dropout=dropout)
        self.conv5 = conv(d_in=n_filters * 2, d_out=n_filters * 2, kernel_size=3, padding=1,
                          stride=1, dilation=1, dropout=dropout)
        self.conv6 = conv(d_in=n_filters * 2, d_out=n_filters * 2, kernel_size=3, padding=1,
                          stride=2, dilation=1, dropout=dropout)
        self.conv7 = conv(d_in=n_filters * 2, d_out=n_filters * 4, kernel_size=3, padding=1,
                          stride=1, dilation=1, dropout=dropout)
        self.conv8 = conv(d_in=n_filters * 4, d_out=n_filters * 4, kernel_size=3, padding=1,
                          stride=1, dilation=1, dropout=dropout)
        self.conv9 = conv(d_in=n_filters * 4, d_out=n_filters * 4, kernel_size=3, padding=1,
                          stride=2, dilation=1, dropout=dropout)

        self.dilate1 = conv(d_in=n_filters * 4, d_out=n_filters * 8, kernel_size=3, padding=1,
                            stride=1, dilation=1, dropout=dropout)
        self.dilate2 = conv(d_in=n_filters * 8, d_out=n_filters * 8, kernel_size=3, padding=2,
                            stride=1, dilation=2, dropout=dropout)
        self.dilate3 = conv(d_in=n_filters * 8, d_out=n_filters * 8, kernel_size=3, padding=4,
                            stride=1, dilation=4, dropout=dropout)
        self.dilate4 = conv(d_in=n_filters * 8, d_out=n_filters * 8, kernel_size=3, padding=8,
                            stride=1, dilation=8, dropout=dropout)
        self.dilate5 = conv(d_in=n_filters * 8, d_out=n_filters * 8, kernel_size=3, padding=16,
                            stride=1, dilation=16, dropout=dropout)
        self.dilate6 = conv(d_in=n_filters * 8, d_out=n_filters * 8, kernel_size=3, padding=32,
                            stride=1, dilation=32, dropout=dropout)

        self.conv10_t = conv_t(d_in=n_filters * 8, d_out=n_filters * 4, kernel_size=3, padding=1,
                               stride=2, dilation=1, dropout=dropout)
        self.conv11 = conv(d_in=n_filters * 4 * 2, d_out=n_filters * 4, kernel_size=3, padding=1,
                           stride=1, dilation=1, dropout=dropout)
        self.conv12_t = conv_t(d_in=n_filters * 4, d_out=n_filters * 2, kernel_size=3, padding=1,
                               stride=2, dilation=1, dropout=dropout)
        self.conv13 = conv(d_in=n_filters * 2 * 2, d_out=n_filters, kernel_size=3, padding=1,
                           stride=1, dilation=1, dropout=dropout)
        self.conv14_t = conv_t(d_in=n_filters, d_out=n_filters, kernel_size=3, padding=1,
                               stride=2, dilation=1, dropout=dropout)
        self.conv15 = conv(d_in=n_filters * 2, d_out=n_filters, kernel_size=3, padding=1,
                           stride=1, dilation=1, dropout=dropout)

    def forward(self, x: tr.Tensor, **kwargs) -> tr.Tensor:
        # x::d_inxHxW
        y_1 = self.conv1(x)
        y_2 = self.conv2(y_1)
        y_3 = self.conv3(y_2)
        y_4 = self.conv4(y_3)
        y_5 = self.conv5(y_4)
        y_6 = self.conv6(y_5)

        y_7 = self.conv7(y_6)
        y_8 = self.conv8(y_7)
        y_9 = self.conv9(y_8)

        y_dilate1 = self.dilate1(y_9)
        y_dilate2 = self.dilate2(y_dilate1)
        y_dilate3 = self.dilate3(y_dilate2)
        y_dilate4 = self.dilate4(y_dilate3)
        y_dilate5 = self.dilate5(y_dilate4)
        y_dilate6 = self.dilate6(y_dilate5)
        y_dilate_sum = y_dilate1 + y_dilate2 + y_dilate3 + y_dilate4 + y_dilate5 + y_dilate6

        y_10 = cat_fn(self.conv10_t(y_dilate_sum), y_7)
        y_11 = self.conv11(y_10)
        y_12 = cat_fn(self.conv12_t(y_11), y_4)
        y_13 = self.conv13(y_12)
        y_14 = cat_fn(self.conv14_t(y_13), y_1)
        y_15 = self.conv15(y_14)
        return y_15

class SafeUAV(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_filters: int):
        super().__init__()
        self.encoder = EncoderMap2Map(in_channels, num_filters)
        self.decoder = nn.Conv2d(in_channels=num_filters, out_channels=out_channels,
                                 kernel_size=1, padding=0, stride=1, dilation=1)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        y_encoder = self.encoder(x)
        y_decoder = self.decoder(y_encoder)
        return y_decoder
