"""
Metrics for five-class retinal artery-vein segmentation.

Dice follows the Sørensen-Dice overlap used as a segmentation objective by
Milletari et al. (2016), DOI: 10.1109/3DV.2016.79. The topology metric follows
clDice by Shit et al. (2021), DOI: 10.1109/CVPR46437.2021.01629. The
presence-conditioned aggregation described below is specific to this study.

Dice convention (5-class):
  Dice is computed PER IMAGE and averaged PRESENCE-CONDITIONED: for each class,
  images whose ground truth does not contain that class are skipped (recorded as
  NaN) instead of scoring a spurious 1.0 (absent/absent) or 0.0. This makes the
  reported Dice independent of batch size and prevents rare classes (overlap,
  ambiguous) from being inflated or deflated by empty images. Aggregate a set of
  per-image records with aggregate_dice_scores().
"""

import math

import numpy as np
import torch

NUM_CLASSES = 5
CLASS_NAMES = {0: "background", 1: "artery", 2: "vein", 3: "overlap", 4: "ambiguous"}
FOREGROUND_IDS = [1, 2, 3, 4]  # non-background classes


def _nanmean(values):
    vals = [v for v in values if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def per_image_multiclass_dice(logits, target, num_classes=NUM_CLASSES, eps=1e-6):
    """
    Per-image, presence-conditioned Dice.

    For each image and class, a class absent from the ground truth is recorded as
    NaN so it is skipped when averaging (see aggregate_dice_scores). Classes that
    are present score the standard 2|A∩B| / (|A|+|B|). Denominator is > 0 whenever
    the class is present, so eps only guards floating-point edge cases.
    """
    pred = torch.argmax(logits, dim=1)
    B = pred.shape[0]
    results = []
    for b in range(B):
        scores = {}
        for c in range(num_classes):
            name = CLASS_NAMES.get(c, f"class_{c}")
            target_c = target[b] == c
            gt_count = target_c.sum().float()
            if gt_count.item() == 0:
                scores[f"dice_{name}"] = float("nan")  # class absent from GT -> skip
                continue
            pred_c = pred[b] == c
            intersection = (pred_c & target_c).sum().float()
            denom = pred_c.sum().float() + gt_count
            scores[f"dice_{name}"] = float((2.0 * intersection) / (denom + eps))

        fg = [scores[f"dice_{CLASS_NAMES[i]}"] for i in FOREGROUND_IDS]
        scores["dice_mean_no_background"] = _nanmean(fg)
        scores["dice_artery_vein_mean"] = _nanmean([scores["dice_artery"], scores["dice_vein"]])
        results.append(scores)
    return results


def aggregate_dice_scores(per_image_records):
    """
    Presence-conditioned mean over a set of per-image records (Dice or topology).

    NaN entries (class absent from that image's GT) are skipped, so each class's
    mean is taken only over images that actually contain the class. Independent of
    how images were grouped into batches.
    """
    if not per_image_records:
        return {}
    keys = list(per_image_records[0].keys())
    return {k: _nanmean([r[k] for r in per_image_records]) for k in keys}


# Topology-aware metrics: clDice + Betti-0 (connectivity)
# clDice source: Shit et al. (2021), DOI: 10.1109/CVPR46437.2021.01629.
# Overlap metrics (Dice/IoU) do not capture whether the vessel tree stays
# connected. clDice (Shit et al., 2021) is the harmonic mean of:
#   Tprec = |skeleton(pred) INSIDE gt|   / |skeleton(pred)|   (no spurious branches)
#   Tsens = |skeleton(gt)   INSIDE pred| / |skeleton(gt)|     (no broken/missed vessels)
# Betti-0 error = |#connected_components(pred) - #components(gt)| measures
# fragmentation. Computed per class (artery, vein), presence-conditioned.


def _num_components(binary):
    try:
        from scipy.ndimage import label
        return int(label(binary)[1])
    except Exception:
        return int(bool(binary.any()))


@torch.no_grad()
def per_image_topology(logits, target, class_ids=(1, 2), eps=1e-6):
    """
    Per-image clDice and Betti-0 error for artery & vein. Presence-conditioned
    (NaN for a class absent from an image's GT). Reuses the Zhang-Suen skeleton.
    NOTE: skeletonisation is on CPU numpy — call only where needed (e.g. the final
    test evaluation), not every training epoch.
    """
    from evaluation.losses.structure_masks import zhang_suen_skeletonize

    pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
    tgt = target.detach().cpu().numpy()
    results = []
    for b in range(pred.shape[0]):
        scores, cldices = {}, []
        for cid in class_ids:
            name = CLASS_NAMES.get(cid, f"class_{cid}")
            g = tgt[b] == cid
            if g.sum() == 0:
                scores[f"cldice_{name}"] = float("nan")
                scores[f"betti0_err_{name}"] = float("nan")
                continue
            p = pred[b] == cid
            sp, sg = zhang_suen_skeletonize(p), zhang_suen_skeletonize(g)
            tprec = (sp & g).sum() / (sp.sum() + eps)     # pred centreline inside GT
            tsens = (sg & p).sum() / (sg.sum() + eps)     # GT centreline inside pred
            cldice = float(2 * tprec * tsens / (tprec + tsens + eps))
            scores[f"cldice_{name}"] = cldice
            scores[f"betti0_err_{name}"] = float(abs(_num_components(p) - _num_components(g)))
            cldices.append(cldice)
        scores["cldice_artery_vein_mean"] = float(np.mean(cldices)) if cldices else float("nan")
        results.append(scores)
    return results


@torch.no_grad()
def confusion_matrix_from_logits(logits, target, num_classes=NUM_CLASSES):
    pred = torch.argmax(logits, dim=1).view(-1).detach().cpu().long()
    target = target.view(-1).detach().cpu().long()
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    encoded = target[valid] * num_classes + pred[valid]
    cm = torch.bincount(encoded, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).numpy()


def pixel_accuracy_from_confusion_matrix(cm):
    total = cm.sum()
    return float(np.trace(cm) / total) if total > 0 else 0.0


@torch.no_grad()
def vessel_density_errors_from_logits(logits, target, num_classes=NUM_CLASSES):
    probs = torch.softmax(logits, dim=1)
    result = {}
    for class_id, name in [(1, "artery"), (2, "vein")]:
        pred_density = probs[:, class_id].mean(dim=(1, 2))
        true_density = (target == class_id).float().mean(dim=(1, 2))
        result[f"density_error_{name}"] = float(torch.abs(pred_density - true_density).mean().item())
    return result


@torch.no_grad()
def local_cbav_region_metrics(logits, target, cbav_masks, eps=1e-6):
    pred = torch.argmax(logits, dim=1)
    out = {}
    for name, mask in cbav_masks.items():
        mask = mask.to(logits.device)
        if mask.ndim == 4: mask = mask[:, 0]
        mask = mask.bool()
        if mask.sum().item() == 0:
            out[f"local_accuracy_{name}"] = 0.0
        else:
            out[f"local_accuracy_{name}"] = float(((pred == target) & mask).sum().float().div(mask.sum().float() + eps).item())
    return out
