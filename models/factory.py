"""Model factory so training/prediction can select architecture by name.

Ordered as an ablation ladder from simplest to most complex:
  basic_unet    - plain double-conv U-Net
  residual_unet - + residual blocks
  attention_unet- + attention gates on skips
  unetpp        - + nested dense skip pathways
  resunetpp     - + residual + squeeze-excite + ASPP + attention
"""

from models.unet import UNet
from models.residual_unet import ResidualUNet
from models.attention_unet import AttentionUNet
from models.unetpp import UNetPP
from models.resunetpp import ResUNetPP

MODEL_CHOICES = ["basic_unet", "residual_unet", "attention_unet", "unetpp", "resunetpp"]

_BUILDERS = {
    "basic_unet": UNet,
    "unet": UNet,
    "residual_unet": ResidualUNet,
    "attention_unet": AttentionUNet,
    "unetpp": UNetPP,
    "unet++": UNetPP,
    "resunetpp": ResUNetPP,
    "resunet++": ResUNetPP,
}


def get_model(name, in_channels=3, out_channels=5, features_start=32):
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"Unknown model: {name}. Choices: {MODEL_CHOICES}")
    return _BUILDERS[key](in_channels=in_channels, out_channels=out_channels,
                          features_start=features_start)
