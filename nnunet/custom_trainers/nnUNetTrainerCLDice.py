"""
Custom nnU-Net v2 trainer for the study's soft-clDice adaptation.

nnU-Net: Isensee et al. (2021), DOI: 10.1038/s41592-020-01008-z;
official code: https://github.com/MIC-DKFZ/nnUNet.
clDice: Shit et al. (2021), DOI: 10.1109/CVPR46437.2021.01629.

The loss used here is CE + Dice + 0.1 * soft-clDice, with `class_weights=None`.
The CE anchor, coefficient and five-class artery/vein mapping are study choices and
are not the reference configuration from the clDice paper.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

PROJECT_ROOT = Path(os.environ.get("RETINA_AV_PROJECT_ROOT", Path.cwd())).resolve()
if not (PROJECT_ROOT / "evaluation" / "losses" / "segmentation_losses.py").is_file():
    raise RuntimeError(
        "Set RETINA_AV_PROJECT_ROOT to the cloned repository before running "
        "the custom nnU-Net trainers."
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from evaluation.losses.segmentation_losses import SoftCLDiceLoss


class _CLDiceLossAdapter(torch.nn.Module):
    """Adapt soft-clDice (DOI: 10.1109/CVPR46437.2021.01629) to nnU-Net targets."""

    def __init__(self, num_classes, cldice_weight=0.1, log_first=4):
        super().__init__()
        self.loss = SoftCLDiceLoss(num_classes=num_classes, class_weights=None,
                                   cldice_weight=cldice_weight, iters=10)
        self._logged = 0
        self._log_first = log_first

    def forward(self, net_output, target):
        if target.ndim == net_output.ndim:      # [B,1,H,W] -> [B,H,W]
            target = target[:, 0]
        target = target.long()
        val = self.loss(net_output, target)
        if self._logged < self._log_first:
            print(f"[CLDicediag] scale {tuple(net_output.shape[-2:])}  loss={float(val):.4f}", flush=True)
            self._logged += 1
        return val


class nnUNetTrainerCLDice(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("soft-clDice trainer assumes label (non-region) segmentation.")
        loss = _CLDiceLossAdapter(self.label_manager.num_segmentation_heads)
        # Mirror nnUNetTrainer._build_loss's deep-supervision wrapping exactly.
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerCLDice_100epochs(nnUNetTrainerCLDice):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100     # matches the CF experiment budget
