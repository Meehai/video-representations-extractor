# pylint: disable=all
import torch as tr
import torch.nn as nn
import torch.nn.functional as F

def conv(dIn, dOut, kernel_size, padding, stride, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels=dIn, out_channels=dOut, kernel_size=kernel_size, padding=padding, \
            stride=stride, dilation=dilation),
        nn.BatchNorm2d(num_features=dOut),
        nn.ReLU(inplace=True)
    )

def conv_tr(dIn, dOut, kernel_size, padding, stride, dilation):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=dIn, out_channels=dOut, kernel_size=kernel_size, padding=padding, \
            stride=stride, dilation=dilation),
        nn.BatchNorm2d(num_features=dOut),
        nn.ReLU(inplace=True)
    )

def fCat(a, b):
    diffUp, diffLeft = b.shape[-2] - a.shape[-2], b.shape[-1] - a.shape[-1]
    a = F.pad(a, (0, diffLeft, 0, diffUp))
    c = tr.cat([a, b], dim=1)
    return c

class EncoderMap2Map(nn.Module):
    def __init__(self, dIn, numFilters=16):
        self.dIn = dIn
        self.numFilters = numFilters
        super().__init__()

        self.conv1 = conv(dIn=dIn, dOut=numFilters, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv2 = conv(dIn=numFilters, dOut=numFilters, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv3 = conv(dIn=numFilters, dOut=numFilters, kernel_size=3, padding=1, stride=2, dilation=1)

        self.conv4 = conv(dIn=numFilters, dOut=numFilters * 2, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv5 = conv(dIn=numFilters * 2, dOut=numFilters * 2, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv6 = conv(dIn=numFilters * 2, dOut=numFilters * 2, kernel_size=3, padding=1, stride=2, dilation=1)

        self.conv7 = conv(dIn=numFilters * 2, dOut=numFilters * 4, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv8 = conv(dIn=numFilters * 4, dOut=numFilters * 4, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv9 = conv(dIn=numFilters * 4, dOut=numFilters * 4, kernel_size=3, padding=1, stride=2, dilation=1)

        self.dilate1 = conv(dIn=numFilters * 4, dOut=numFilters * 8, kernel_size=3, padding=1, stride=1, dilation=1)
        self.dilate2 = conv(dIn=numFilters * 8, dOut=numFilters * 8, kernel_size=3, padding=2, stride=1, dilation=2)
        self.dilate3 = conv(dIn=numFilters * 8, dOut=numFilters * 8, kernel_size=3, padding=4, stride=1, dilation=4)
        self.dilate4 = conv(dIn=numFilters * 8, dOut=numFilters * 8, kernel_size=3, padding=8, stride=1, dilation=8)
        self.dilate5 = conv(dIn=numFilters * 8, dOut=numFilters * 8, kernel_size=3, padding=16, stride=1, dilation=16)
        self.dilate6 = conv(dIn=numFilters * 8, dOut=numFilters * 8, kernel_size=3, padding=32, stride=1, dilation=32)

        self.conv_transpose10 = conv_tr(dIn=numFilters * 8, dOut=numFilters * 4, kernel_size=3, \
            padding=1, stride=2, dilation=1)
        self.conv11 = conv(dIn=numFilters * 4 * 2, dOut=numFilters * 4, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv_transpose12 = conv_tr(dIn=numFilters * 4, dOut=numFilters * 2, kernel_size=3, \
            padding=1, stride=2, dilation=1)
        self.conv13 = conv(dIn=numFilters * 2 * 2, dOut=numFilters, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv_transpose14 = conv_tr(dIn=numFilters, dOut=numFilters, kernel_size=3, \
            padding=1, stride=2, dilation=1)
        self.conv15 = conv(dIn=numFilters * 2, dOut=numFilters, kernel_size=3, padding=1, stride=1, dilation=1)

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        # x::dInxHxW
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

        y_10 = fCat(self.conv_transpose10(y_dilate_sum), y_7)
        y_11 = self.conv11(y_10)
        y_12 = fCat(self.conv_transpose12(y_11), y_4)
        y_13 = self.conv13(y_12)
        y_14 = fCat(self.conv_transpose14(y_13), y_1)
        y_15 = self.conv15(y_14)
        return y_15

    def __str__(self):
        return f"Encoder Map2Map. dIn: {self.dIn}. NF: {self.numFilters}."

class DecoderMap2Map(nn.Module):
    def __init__(self, dOut, numFilters=16):
        self.dOut = dOut
        self.numFilters = numFilters
        super().__init__()

        self.decoderConv = nn.Conv2d(in_channels=numFilters, out_channels=dOut, kernel_size=1, \
            padding=0, stride=1, dilation=1)

    def forward(self, x):
        y = self.decoderConv(x)
        return y

    def __str__(self):
        return f"Decoder Map2Map. dOut: {self.dOut}. NF: {self.numFilters}."
