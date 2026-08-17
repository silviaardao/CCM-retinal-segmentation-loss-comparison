"""
U-Net++ / Nested U-Net (Zhou et al., 2018,
DOI: 10.1007/978-3-030-00889-5_1): a U-Net whose skip connections are
re-engineered into dense nested pathways. Each node X[i][j] fuses all shallower
same-resolution nodes with the up-sampled node from one level below. Isolates the
effect of nested dense skips. This is a study implementation, not the authors'
reference code.

Same interface as unet.py: (in_channels, out_channels, features_start),
returns raw logits [B, out_channels, H, W].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import DoubleConv


class UNetPP(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features_start=32):
        super().__init__()
        f = int(features_start)
        ch = [f, f * 2, f * 4, f * 8, f * 16]   # channels per row i
        self.depth = len(ch)                     # 5 rows -> j up to 4
        self.pool = nn.MaxPool2d(2)

        # Backbone encoder column j=0.
        self.blocks = nn.ModuleDict()
        for i in range(self.depth):
            in_c = in_channels if i == 0 else ch[i - 1]
            self.blocks[f"x{i}_0"] = DoubleConv(in_c, ch[i])

        # Nested decoder nodes X[i][j], j>=1.
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                # j same-row shallower nodes (each ch[i]) + up(X[i+1][j-1]) (ch[i+1]).
                in_c = ch[i] * j + ch[i + 1]
                self.blocks[f"x{i}_{j}"] = DoubleConv(in_c, ch[i])

        self.outc = nn.Conv2d(ch[0], out_channels, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, x):
        X = {}
        # Encoder column.
        for i in range(self.depth):
            inp = x if i == 0 else self.pool(X[(i - 1, 0)])
            X[(i, 0)] = self.blocks[f"x{i}_0"](inp)

        # Nested nodes.
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                skips = [X[(i, k)] for k in range(j)]
                up = self._up(X[(i + 1, j - 1)], X[(i, 0)])
                X[(i, j)] = self.blocks[f"x{i}_{j}"](torch.cat(skips + [up], dim=1))

        return self.outc(X[(0, self.depth - 1)])
