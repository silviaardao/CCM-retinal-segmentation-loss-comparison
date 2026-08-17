"""
Custom nnU-Net v2 trainers for the study's CF-Loss adaptation.

nnU-Net: Isensee et al. (2021), DOI: 10.1038/s41592-020-01008-z;
official code: https://github.com/MIC-DKFZ/nnUNet.
CF-Loss: Zhou et al. (2024), DOI: 10.1016/j.media.2024.103098.

These subclasses replace the loss construction while retaining nnU-Net's configured
network, preprocessing, augmentation, optimization schedule and deep supervision.
Select them with `-tr nnUNetTrainerCFv`, `nnUNetTrainerCFb`, or
`nnUNetTrainerCFvb` (and the corresponding 100-epoch variants).

This file is the source of truth. A copy is placed under the venv's
training/nnUNetTrainer/variants/loss/ folder so nnU-Net's class discovery finds
it; keep the two in sync (scripts/install_nnunet_custom_trainers.py copies it there).

The adapter accepts nnU-Net's target tensor shape and applies the same
deep-supervision weighting pattern as the installed nnU-Net version. CF-Loss itself
is implemented in `evaluation/losses/segmentation_losses.py`, including the study-specific
five-class mapping. The feature weights follow the source configuration:
vessel-density beta=1.0 and box-count lambda=0.5. The CE base is unweighted in
this nnU-Net experiment (`class_weights=None`).
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
from evaluation.losses.segmentation_losses import CFLoss


class _CFLossAdapter(torch.nn.Module):
    """Adapt CF-Loss (DOI: 10.1016/j.media.2024.103098) to nnU-Net targets."""

    def __init__(self, num_classes, mode, log_first=4):
        super().__init__()
        # Source coefficients; mode selects which feature terms are active.
        self.cf = CFLoss(num_classes=num_classes, mode=mode,
                         beta_vessel_density=1.0, lambda_box_count=0.5)
        self._logged = 0
        self._log_first = log_first

    def forward(self, net_output, target):
        if target.ndim == net_output.ndim:      # [B,1,H,W] -> [B,H,W]
            target = target[:, 0]
        target = target.long()
        if self._logged < self._log_first:       # diagnose the terms on the first calls
            with torch.no_grad():
                ce = self.cf.base_ce(net_output, target).item()
                vd = self.cf.vessel_density(net_output, target).item() if "v" in self.cf.mode else 0.0
                bc = self.cf.box_count(net_output, target).item() if "b" in self.cf.mode else 0.0
                print(f"[CFdiag] scale {tuple(net_output.shape[-2:])}  "
                      f"CE={ce:.4f}  Vdens={vd:.4f}  Box={bc:.4f}", flush=True)
            self._logged += 1
        return self.cf(net_output, target)


class _CFTrainerBase(nnUNetTrainer):
    CF_MODE = "vb"

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("CF-Loss trainer assumes label (non-region) segmentation.")
        loss = _CFLossAdapter(self.label_manager.num_segmentation_heads, self.CF_MODE)
        # Mirror nnUNetTrainer._build_loss's deep-supervision wrapping exactly.
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerCFv(_CFTrainerBase):
    CF_MODE = "v"       # vessel density only


class nnUNetTrainerCFb(_CFTrainerBase):
    CF_MODE = "b"       # box-count (fractal) only


class nnUNetTrainerCFvb(_CFTrainerBase):
    CF_MODE = "vb"      # both


# Short sanity variant
class nnUNetTrainerCFvb_2epochs(nnUNetTrainerCFvb):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2


# 100-epoch variants
_EPOCHS = 100


class nnUNetTrainerCFv_100epochs(nnUNetTrainerCFv):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = _EPOCHS


class nnUNetTrainerCFb_100epochs(nnUNetTrainerCFb):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = _EPOCHS


class nnUNetTrainerCFvb_100epochs(nnUNetTrainerCFvb):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = _EPOCHS


# Generic-loss controls at the same 100-epoch budget. Dice and CE subclass
# nnU-Net's own loss trainers; the Dice+CE default is the built-in
# nnUNetTrainer_100epochs (no subclass needed).
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceLoss
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerCELoss import nnUNetTrainerCELoss


class nnUNetTrainerDiceLoss_100epochs(nnUNetTrainerDiceLoss):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = _EPOCHS


class nnUNetTrainerCELoss_100epochs(nnUNetTrainerCELoss):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = _EPOCHS
