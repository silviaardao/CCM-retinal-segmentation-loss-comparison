"""
Residual U-Net adaptation based on Zhang, Liu and Wang (2018),
DOI: 10.1109/LGRS.2018.2802944. The basic U-Net double convolution is replaced by a residual
unit (conv-BN-ReLU x2 + identity/1x1 shortcut). This isolates the effect of
adding residual connections on top of the plain U-Net. This is a study implementation,
not the authors' reference code.

Same interface as unet.py: (in_channels, out_channels, features_start),
returns raw logits [B, out_channels, H, W].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.main(x) + self.shortcut(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), ResidualConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ResidualConv(in_channels, out_channels)

    def forward(self, x_decoder, x_encoder):
        x_decoder = self.up(x_decoder)
        diff_y = x_encoder.size(2) - x_decoder.size(2)
        diff_x = x_encoder.size(3) - x_decoder.size(3)
        x_decoder = F.pad(x_decoder, [diff_x // 2, diff_x - diff_x // 2,
                                      diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x_encoder, x_decoder], dim=1))


class ResidualUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features_start=32):
        super().__init__()
        f = int(features_start)
        self.inc = ResidualConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)
        self.up1 = Up(f * 16 + f * 8, f * 8)
        self.up2 = Up(f * 8 + f * 4, f * 4)
        self.up3 = Up(f * 4 + f * 2, f * 2)
        self.up4 = Up(f * 2 + f, f)
        self.outc = nn.Conv2d(f, out_channels, 1)

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
