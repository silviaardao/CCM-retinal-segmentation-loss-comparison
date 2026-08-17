"""
Attention U-Net (Oktay et al., 2018, https://arxiv.org/abs/1804.03999):
the basic U-Net with additive attention
gates on the skip connections. The decoder feature gates the encoder skip so the
network learns where to focus before concatenation. Isolates the effect of adding
attention (no residual blocks).

Same interface as unet.py: (in_channels, out_channels, features_start),
returns raw logits [B, out_channels, H, W].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import DoubleConv


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018). Gates skip x with gating signal g."""

    def __init__(self, x_channels, g_channels, inter_channels):
        super().__init__()
        self.w_x = nn.Sequential(nn.Conv2d(x_channels, inter_channels, 1), nn.BatchNorm2d(inter_channels))
        self.w_g = nn.Sequential(nn.Conv2d(g_channels, inter_channels, 1), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x, g):
        att = self.psi(F.relu(self.w_x(x) + self.w_g(g)))
        return x * att


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class AttentionUp(nn.Module):
    def __init__(self, decoder_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.att = AttentionGate(skip_channels, decoder_channels, max(1, skip_channels // 2))
        self.conv = DoubleConv(decoder_channels + skip_channels, out_channels)

    def forward(self, x_decoder, x_encoder):
        x_decoder = self.up(x_decoder)
        diff_y = x_encoder.size(2) - x_decoder.size(2)
        diff_x = x_encoder.size(3) - x_decoder.size(3)
        x_decoder = F.pad(x_decoder, [diff_x // 2, diff_x - diff_x // 2,
                                      diff_y // 2, diff_y - diff_y // 2])
        x_encoder = self.att(x_encoder, x_decoder)          # gate the skip
        return self.conv(torch.cat([x_encoder, x_decoder], dim=1))


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features_start=32):
        super().__init__()
        f = int(features_start)
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)
        self.up1 = AttentionUp(f * 16, f * 8, f * 8)
        self.up2 = AttentionUp(f * 8, f * 4, f * 4)
        self.up3 = AttentionUp(f * 4, f * 2, f * 2)
        self.up4 = AttentionUp(f * 2, f, f)
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
