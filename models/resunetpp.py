"""
ResUNet++ (Jha et al., 2019, "ResUNet++: An Advanced Architecture for Medical
Image Segmentation", https://arxiv.org/abs/1911.07067). Study-specific PyTorch
implementation of the published architecture:

  - Stem: residual conv block
  - Encoder: squeeze-excite -> residual unit with strided downsampling (x3)
  - Bridge: ASPP (atrous spatial pyramid pooling)
  - Decoder: attention gate -> upsample -> concat skip -> residual unit (x3)
  - Output: ASPP -> 1x1 conv

Same interface unet.py: (in_channels, out_channels, features_start),
returns raw logits [B, out_channels, H, W]. Do not apply softmax inside.

Minor implementation details, including ASPP rates and the attention-gate form, may
differ from the authors' reference implementation; exact implementation parity is
not claimed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ResidualBlock(nn.Module):
    """Pre-activation residual unit; stride>1 downsamples in the first conv."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r),
                nn.BatchNorm2d(out_channels),
            ) for r in rates
        ])
        self.project = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        y = sum(branch(x) for branch in self.branches)
        return self.project(F.relu(y))


class AttentionGate(nn.Module):
    """Additive attention gate: gates the skip feature x with the decoder signal g."""

    def __init__(self, x_channels, g_channels, inter_channels):
        super().__init__()
        self.theta_x = nn.Conv2d(x_channels, inter_channels, 1)
        self.phi_g = nn.Conv2d(g_channels, inter_channels, 1)
        self.psi = nn.Conv2d(inter_channels, 1, 1)

    def forward(self, x, g):
        att = torch.sigmoid(self.psi(F.relu(self.theta_x(x) + self.phi_g(g))))
        return x * att


class ResUNetPP(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features_start=32):
        super().__init__()
        f = int(features_start)

        self.stem = Stem(in_channels, f)
        self.se1 = SqueezeExcite(f)
        self.enc1 = ResidualBlock(f, f * 2, stride=2)
        self.se2 = SqueezeExcite(f * 2)
        self.enc2 = ResidualBlock(f * 2, f * 4, stride=2)
        self.se3 = SqueezeExcite(f * 4)
        self.enc3 = ResidualBlock(f * 4, f * 8, stride=2)

        self.bridge = ASPP(f * 8, f * 16)

        self.att3 = AttentionGate(f * 4, f * 16, f * 4)
        self.dec3 = ResidualBlock(f * 16 + f * 4, f * 8)
        self.att2 = AttentionGate(f * 2, f * 8, f * 2)
        self.dec2 = ResidualBlock(f * 8 + f * 2, f * 4)
        self.att1 = AttentionGate(f, f * 4, f)
        self.dec1 = ResidualBlock(f * 4 + f, f * 2)

        self.out_aspp = ASPP(f * 2, f)
        self.outc = nn.Conv2d(f, out_channels, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, x):
        c1 = self.stem(x)                 # f,      H
        c2 = self.enc1(self.se1(c1))      # 2f,     H/2
        c3 = self.enc2(self.se2(c2))      # 4f,     H/4
        c4 = self.enc3(self.se3(c3))      # 8f,     H/8

        b = self.bridge(c4)               # 16f,    H/8

        g3 = self._up(b, c3)
        d3 = self.dec3(torch.cat([self.att3(c3, g3), g3], dim=1))   # 8f, H/4
        g2 = self._up(d3, c2)
        d2 = self.dec2(torch.cat([self.att2(c2, g2), g2], dim=1))   # 4f, H/2
        g1 = self._up(d2, c1)
        d1 = self.dec1(torch.cat([self.att1(c1, g1), g1], dim=1))   # 2f, H

        return self.outc(self.out_aspp(d1))
