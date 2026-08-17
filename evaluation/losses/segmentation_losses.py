"""
Loss functions for five-class retinal artery-vein segmentation.

Each non-trivial loss cites its source below. The implementations are study-specific
PyTorch implementations; CF-Loss, CBAV-Loss and soft-clDice are adapted to the common
five-class artery/vein label scheme and should not be treated as authors' reference code.

  - Dice: Milletari et al., V-Net, 3DV 2016. DOI: 10.1109/3DV.2016.79.
  - Focal loss: Lin et al., ICCV 2017. DOI: 10.1109/ICCV.2017.324.
  - CFLoss (cf_v/cf_b/cf_vb) — Zhou et al., "CF-Loss…", Medical Image Analysis, 2024:
      DOI: 10.1016/j.media.2024.103098.
      base = cross-entropy (Eq 7-9); VesselDensityLoss = Eq 2; SoftBoxCountLoss = Eq 6
      (second soft box count, box-size-weighted L2).
  - CBAVLoss (cbav) — Zhang et al., "CBAV-Loss…", PRCV 2023:
      DOI: 10.1007/978-981-99-8558-6_5.
      base = 1.0*CE + 0.6*Dice; local crossover/branch error via Cov(P-G, R) (Eq 1-4);
      lambda_cross = lambda_branch = 0.005; r_c = 8, r_b = 5.  Fundus 5-class adaptation of
      the original OCTA method.
  - SoftCLDiceLoss (cldice) — Shit et al., CVPR 2021.
      DOI: 10.1109/CVPR46437.2021.01629. Soft-skeleton clDice + Dice;
      here `weighted_CE + Dice + 0.1*soft-clDice` (CE anchor added to prevent collapse).

The architecture files in models/ contain their corresponding references.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.losses.structure_masks import cbav_point_masks

# Multiclass components

NUM_CLASSES = 5


class MulticlassDiceLoss(nn.Module):
    """Multiclass soft Dice adapted from Milletari et al., DOI: 10.1109/3DV.2016.79."""
    def __init__(self, num_classes=NUM_CLASSES, include_background=False, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dims)
        cardinality = torch.sum(probs + target_oh, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    """CE + Dice. Used as BASELINE loss and as CBAV base. NOT used inside CF-Loss."""

    def __init__(self, num_classes=NUM_CLASSES, class_weights=None, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = MulticlassDiceLoss(num_classes=num_classes, include_background=False)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


class MulticlassFocalLoss(nn.Module):
    """
    Multiclass focal loss (Lin et al., 2017; DOI: 10.1109/ICCV.2017.324):
    alpha_t * (1 - p_t)^gamma * (-log p_t).

    The modulating factor uses the UNWEIGHTED p_t = exp(-CE), so class weights act
    as the alpha_t balancing term only and do not distort (1 - p_t)^gamma. (An
    earlier version folded the weight into CE before exp(), which made the factor
    (1 - p_t^w)^gamma and departed from the paper definition.)
    """

    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")  # unweighted: -log p_t
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.class_weights is not None:
            alpha_t = self.class_weights.to(logits.device)[target]  # per-pixel alpha
            focal = alpha_t * focal
        return focal.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, gamma=2.0, class_weights=None):
        super().__init__()
        self.focal = MulticlassFocalLoss(gamma=gamma, class_weights=class_weights)
        self.dice = MulticlassDiceLoss(num_classes=num_classes, include_background=False)

    def forward(self, logits, target):
        return self.focal(logits, target) + self.dice(logits, target)

# CF-Loss (Zhou et al., MedIA 2024)
# Base = CE only, per paper Equations 7-9

class VesselDensityLoss(nn.Module):
    """CF-Loss Eq. 2: L1 error in artery/vein area fractions.

    Zhou et al. (2024), DOI: 10.1016/j.media.2024.103098.
    """

    def __init__(self, num_classes=NUM_CLASSES, class_ids=(1, 2)):
        super().__init__()
        self.num_classes = num_classes
        self.class_ids = class_ids

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        loss = logits.new_tensor(0.0)
        for cid in self.class_ids:
            loss = loss + torch.abs(probs[:, cid].mean(dim=(1, 2)) - target_oh[:, cid].mean(dim=(1, 2))).mean()
        return loss


class SoftBoxCountLoss(nn.Module):
    """
    CF-Loss Eq. 6: weighted L2 of per-scale normalised box-count errors.
    Uses second soft box count (per-box L1 distances).
    Box sizes = powers of 2.

    Zhou et al. (2024), DOI: 10.1016/j.media.2024.103098.
    """

    def __init__(self, num_classes=NUM_CLASSES, class_ids=(1, 2), eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.class_ids = class_ids
        self.eps = eps

    @staticmethod
    def _pad_to_multiple(x, k):
        h, w = x.shape[-2:]
        pad_h = (k - h % k) % k
        pad_w = (k - w % k) % k
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        h, w = logits.shape[-2:]

        box_sizes = []
        k = 2
        while k <= min(h, w):
            box_sizes.append(k)
            k *= 2

        if not box_sizes:
            return logits.new_tensor(0.0)

        norm = 1.0 / (sum(float(k) ** 2 for k in box_sizes) ** 0.5 + self.eps)
        total_loss = logits.new_tensor(0.0)

        for cid in self.class_ids:
            pred = probs[:, cid:cid + 1]
            true = target_oh[:, cid:cid + 1]
            weighted_sq_sum = logits.new_tensor(0.0)

            for k in box_sizes:
                pred_pool = F.avg_pool2d(self._pad_to_multiple(pred, k), kernel_size=k, stride=k)
                true_pool = F.avg_pool2d(self._pad_to_multiple(true, k), kernel_size=k, stride=k)

                n_pred = torch.abs(pred_pool - true_pool).sum(dim=(1, 2, 3))
                n_true = true_pool.sum(dim=(1, 2, 3))
                normalised_error = n_pred / (n_true + self.eps)
                weighted_sq_sum = weighted_sq_sum + float(k) * (normalised_error ** 2).mean()

            total_loss = total_loss + norm * torch.sqrt(weighted_sq_sum + self.eps)

        return total_loss


class CFLoss(nn.Module):
    """
    CF-Loss (Zhou et al., Medical Image Analysis 2024), Eqs. 7-9.
    DOI: 10.1016/j.media.2024.103098. Base = CE only (not CE+Dice).
    `lambda_box_count` weights L_B (fractal/box-count), `beta_vessel_density` weights L_V.

    The coefficients follow the source configuration: box-count lambda=0.5 and
    vessel-density beta=1.0.
    """

    def __init__(self, num_classes=NUM_CLASSES, class_weights=None, mode="vb",
                 lambda_box_count=0.5, beta_vessel_density=1.0):
        super().__init__()
        self.mode = mode
        self.base_ce = nn.CrossEntropyLoss(weight=class_weights)
        self.vessel_density = VesselDensityLoss(num_classes=num_classes, class_ids=(1, 2))
        self.box_count = SoftBoxCountLoss(num_classes=num_classes, class_ids=(1, 2))
        self.lambda_box_count = lambda_box_count
        self.beta_vessel_density = beta_vessel_density

    def forward(self, logits, target):
        loss = self.base_ce(logits, target)
        if "v" in self.mode:
            loss = loss + self.beta_vessel_density * self.vessel_density(logits, target)
        if "b" in self.mode:
            loss = loss + self.lambda_box_count * self.box_count(logits, target)
        return loss


# CBAV-Loss (Zhang et al., PRCV 2023), DOI: 10.1007/978-981-99-8558-6_5
#
# Paper (§2.2-2.3):
#   D_{a,r}  = Cov(P_a - G_a, R_r)          local sum of the signed artery
#                                           over/under-segmentation error in a
#                                           (1+2r)x(1+2r) neighbourhood (Eq 3)
#   Lcross   = ||Mcross . D_{a,rc}|| / (rc^2 ||Mcross||)
#            + ||Mcross . D_{v,rc}|| / (rc^2 ||Mcross||)                    (Eq 1)
#   Lbranch  = ||MbranchA . D_{a,rb}|| / (rb^2 ||MbranchA||)
#            + ||MbranchV . D_{v,rb}|| / (rb^2 ||MbranchV||)                (Eq 2)
#   Ltotal   = 1.0*CE + 0.6*Dice + 0.005*Lcross + 0.005*Lbranch            (Eq 4)
#   rc = 8, rb = 5.  Mcross = SA n SV; Mbranch = skeleton pts w/ >=3 neighbours.
#
# Fundus adaptation (paper used OCTA, 2-class): overlap (class 3) counts as both
# artery and vein when forming G_a/G_v and the skeletons (see structure_masks).
# Structure masks are derived from the (augmented) target on the fly, so they stay
# aligned with augmented images and no precomputed masks are needed.


class CBAVLoss(nn.Module):
    """Five-class fundus adaptation of CBAV-Loss Eqs. 1-4.

    Zhang et al. (2023), DOI: 10.1007/978-981-99-8558-6_5. The source method was
    proposed for OCTA; this implementation folds the fundus overlap class into both
    artery and vein masks when constructing crossover and branch terms.
    """
    def __init__(self, num_classes=NUM_CLASSES, class_weights=None,
                 lambda_cross=0.005, lambda_branch=0.005, rc=8, rb=5, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        # Paper base loss Lb = 1.0*CE + 0.6*Dice.
        self.base = CEDiceLoss(num_classes=num_classes, class_weights=class_weights,
                               ce_weight=1.0, dice_weight=0.6)
        self.lambda_cross = lambda_cross
        self.lambda_branch = lambda_branch
        self.rc = rc
        self.rb = rb
        self.eps = eps
        # Fixed all-ones box kernels R_rc, R_rb for the Cov(.) neighbourhood sum.
        self.register_buffer("kernel_c", torch.ones(1, 1, 1 + 2 * rc, 1 + 2 * rc))
        self.register_buffer("kernel_b", torch.ones(1, 1, 1 + 2 * rb, 1 + 2 * rb))

    def _build_masks(self, target):
        """Per-image (Mcross, MbranchA, MbranchV) from the GT label -> [B,1,H,W]."""
        B, H, W = target.shape
        cross = np.zeros((B, 1, H, W), dtype=np.float32)
        branch_a = np.zeros((B, 1, H, W), dtype=np.float32)
        branch_v = np.zeros((B, 1, H, W), dtype=np.float32)
        tnp = target.detach().cpu().numpy()
        for b in range(B):
            mc, mba, mbv = cbav_point_masks(tnp[b])
            cross[b, 0] = mc
            branch_a[b, 0] = mba
            branch_v[b, 0] = mbv
        dev = target.device
        return (torch.from_numpy(cross).to(dev),
                torch.from_numpy(branch_a).to(dev),
                torch.from_numpy(branch_v).to(dev))

    def _local_term(self, error, mask, kernel, r):
        """||mask . Cov(error, R_r)|| / (r^2 ||mask||), summed per image -> [B]."""
        d = F.conv2d(error, kernel.to(error.device), padding=r)   # Cov(error, R_r)
        num = (mask * d.abs()).sum(dim=(1, 2, 3))
        denom = (r * r) * mask.sum(dim=(1, 2, 3)) + self.eps
        return num / denom

    def forward(self, logits, target):
        loss = self.base(logits, target)
        probs = torch.softmax(logits, dim=1)

        # Overlap (3) is physically both artery and vein at a crossing.
        p_a = probs[:, 1] + probs[:, 3]
        p_v = probs[:, 2] + probs[:, 3]
        g_a = ((target == 1) | (target == 3)).float()
        g_v = ((target == 2) | (target == 3)).float()
        err_a = (p_a - g_a).unsqueeze(1)                  # [B,1,H,W]
        err_v = (p_v - g_v).unsqueeze(1)

        with torch.no_grad():
            m_cross, m_branch_a, m_branch_v = self._build_masks(target)

        l_cross = (self._local_term(err_a, m_cross, self.kernel_c, self.rc)
                   + self._local_term(err_v, m_cross, self.kernel_c, self.rc)).mean()
        l_branch = (self._local_term(err_a, m_branch_a, self.kernel_b, self.rb)
                    + self._local_term(err_v, m_branch_v, self.kernel_b, self.rb)).mean()

        return loss + self.lambda_cross * l_cross + self.lambda_branch * l_branch


# Soft-clDice

class SoftCLDiceLoss(nn.Module):
    """Multiclass adaptation of soft-clDice.

    Shit et al. (2021), DOI: 10.1109/CVPR46437.2021.01629. Combines a pixelwise
    CE+Dice base with a differentiable soft-clDice term on the vessel classes
    (artery/vein, overlap folded into BOTH trees, matching this study's clDice metric):

        L = (1 - alpha) * softDice  +  alpha * softclDice

    Soft skeletonisation is iterative min-pool (erode) / max-pool (dilate), so the whole
    operation remains differentiable. The CE anchor and the 0.1 weighting are study
    choices rather than settings specified by the clDice paper."""

    def __init__(self, num_classes=NUM_CLASSES, class_weights=None, cldice_weight=0.1,
                 iters=10, smooth=1.0):
        super().__init__()
        # Stable base (weighted CE + Dice) grounds the pixels; a SMALL clDice term adds
        # centreline topology without destabilising. A pure softDice+softclDice loss (no CE,
        # high weight) collapses from scratch — the skeleton of early near-uniform predictions
        # is meaningless, so clDice needs a CE anchor and a modest weight.
        self.base = CEDiceLoss(num_classes=num_classes, class_weights=class_weights)
        self.cldice_weight = float(cldice_weight)
        self.iters = int(iters)
        self.smooth = float(smooth)

    @staticmethod
    def _erode(x):
        p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    @staticmethod
    def _dilate(x):
        return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))

    def _open(self, x):
        return self._dilate(self._erode(x))

    def _skeletonize(self, x):
        x1 = self._open(x)
        skel = F.relu(x - x1)
        for _ in range(self.iters):
            x = self._erode(x)
            x1 = self._open(x)
            delta = F.relu(x - x1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def _cldice(self, p, t):
        sp, st = self._skeletonize(p), self._skeletonize(t)
        tprec = (torch.sum(sp * t) + self.smooth) / (torch.sum(sp) + self.smooth)
        tsens = (torch.sum(st * p) + self.smooth) / (torch.sum(st) + self.smooth)
        return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        # overlap (class 3) folded into both artery (1) and vein (2)
        p_art = (probs[:, 1] + probs[:, 3]).clamp(0, 1).unsqueeze(1)
        p_vein = (probs[:, 2] + probs[:, 3]).clamp(0, 1).unsqueeze(1)
        t_art = ((target == 1) | (target == 3)).float().unsqueeze(1)
        t_vein = ((target == 2) | (target == 3)).float().unsqueeze(1)
        cl = 0.5 * (self._cldice(p_art, t_art) + self._cldice(p_vein, t_vein))
        return self.base(logits, target) + self.cldice_weight * cl


# Factory

def get_multiclass_loss(loss_name, num_classes=NUM_CLASSES, class_weights=None):
    loss_name = loss_name.lower()

    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_name == "cldice":
        return SoftCLDiceLoss(num_classes=num_classes, class_weights=class_weights,
                              cldice_weight=0.1, iters=10)
    if loss_name == "dice":
        return MulticlassDiceLoss(num_classes=num_classes, include_background=False)
    if loss_name == "ce_dice":
        return CEDiceLoss(num_classes=num_classes, class_weights=None)
    if loss_name == "weighted_ce_dice":
        return CEDiceLoss(num_classes=num_classes, class_weights=class_weights)
    if loss_name == "focal_dice":
        return FocalDiceLoss(num_classes=num_classes, gamma=2.0, class_weights=class_weights)
    if loss_name == "cf_v":
        return CFLoss(num_classes=num_classes, class_weights=class_weights,
                      mode="v", beta_vessel_density=1.0, lambda_box_count=0.0)
    if loss_name == "cf_b":
        return CFLoss(num_classes=num_classes, class_weights=class_weights,
                      mode="b", beta_vessel_density=0.0, lambda_box_count=0.5)
    if loss_name == "cf_vb":
        return CFLoss(num_classes=num_classes, class_weights=class_weights,
                      mode="vb", beta_vessel_density=1.0, lambda_box_count=0.5)
    if loss_name == "cbav":
        return CBAVLoss(num_classes=num_classes, class_weights=class_weights,
                        lambda_cross=0.005, lambda_branch=0.005)
    raise ValueError(f"Unknown multiclass loss: {loss_name}")
