"""U-Net implementation for the common study interface.

Ronneberger, Fischer and Brox (2015), "U-Net: Convolutional Networks for
Biomedical Image Segmentation", DOI: 10.1007/978-3-319-24574-4_28.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsample -> concatenate skip connection -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder, x_encoder):
        x_decoder = self.up(x_decoder)

        diff_y = x_encoder.size(2) - x_decoder.size(2)
        diff_x = x_encoder.size(3) - x_decoder.size(3)
        x_decoder = F.pad(
            x_decoder,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x_encoder, x_decoder], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Basic 2D U-Net.

    Binary segmentation: out_channels=1, use BCEWithLogits-style loss.
    Multiclass segmentation: out_channels=num_classes, use CrossEntropyLoss-style loss.

    The model returns raw logits. Do not apply sigmoid/softmax inside the model.
    """

    def __init__(self, in_channels=3, out_channels=1, features_start=32):
        super().__init__()
        f = int(features_start)

        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)

        self.up1 = Up(f * 16 + f * 8, f * 8)
        self.up2 = Up(f * 8 + f * 4, f * 4)
        self.up3 = Up(f * 4 + f * 2, f * 2)
        self.up4 = Up(f * 2 + f, f)

        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
