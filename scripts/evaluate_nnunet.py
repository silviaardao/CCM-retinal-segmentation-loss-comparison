"""
Score nnU-Net's held-out test predictions with the study's evaluation metrics.

Reads predicted label PNGs (0..4) and the matching ground-truth label PNGs from
the nnU-Net dataset's labelsTs, computes per-image presence-conditioned Dice
(identical convention to our training pipeline), and writes a metrics JSON + CSV.
"""

from pathlib import Path
import sys
import argparse
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from evaluation.metrics.segmentation_metrics import (
    per_image_multiclass_dice, per_image_topology, aggregate_dice_scores,
)

NUM_CLASSES = 5
EVAL_SIZE = 512   # match the model pipeline's eval resolution for a fair comparison


def label_to_fake_logits(pred):
    """One-hot the predicted class map so argmax recovers it inside the metric."""
    t = torch.from_numpy(pred.astype(np.int64)).unsqueeze(0)          # [1,H,W]
    return F.one_hot(t, NUM_CLASSES).permute(0, 3, 1, 2).float() * 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(PROJECT_ROOT / "nnunet" / "predictions_test"))
    ap.add_argument("--gt-dir",
                    default=str(PROJECT_ROOT / "nnunet" / "nnUNet_raw" /
                               "Dataset001_RetinaAV" / "labelsTs"))
    args = ap.parse_args()

    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    preds = sorted(pred_dir.glob("*.png"))
    if not preds:
        raise SystemExit(f"No predictions in {pred_dir}. Run nnUNetv2_predict first.")

    def to_eval_size(arr):
        return np.array(Image.fromarray(arr.astype(np.uint8)).resize(
            (EVAL_SIZE, EVAL_SIZE), Image.NEAREST))

    records = []
    for p in preds:
        gt_p = gt_dir / p.name
        if not gt_p.exists():
            print(f"  (skip {p.name}: no GT)")
            continue
        pred = to_eval_size(np.array(Image.open(p)))     # both at 512, like our model eval
        gt = to_eval_size(np.array(Image.open(gt_p)))
        logits = label_to_fake_logits(pred)
        target = torch.from_numpy(gt.astype(np.int64)).unsqueeze(0)
        dice_rec = per_image_multiclass_dice(logits, target, num_classes=NUM_CLASSES)[0]
        topo_rec = per_image_topology(logits, target)[0]
        # Record the gid. This loop runs in sorted-FILENAME order, which is NOT the
        # manifest order our own results_per_image.csv uses (they diverge from
        # les_av/fundus_avseg onwards), so image_idx is not comparable across the
        # two tables - anything pairing these per-image MUST join on gid.
        records.append({"gid": p.stem, **dice_rec, **topo_rec})

    # aggregate over the numeric fields only (gid is bookkeeping, not a metric)
    agg = aggregate_dice_scores([{k: v for k, v in r.items() if k != "gid"} for r in records])
    out_json = PROJECT_ROOT / "nnunet" / "nnunet_test_metrics.json"
    with open(out_json, "w") as f:
        json.dump(agg, f, indent=4)
    with open(PROJECT_ROOT / "nnunet" / "nnunet_test_per_image_dice.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_idx"] + list(records[0].keys()))
        # image_idx is kept only for backwards compatibility; join on gid.
        w.writeheader()
        for i, r in enumerate(records):
            w.writerow({"image_idx": i, **r})

    print(f"nnU-Net test metrics ({len(records)} images) -> {out_json}")
    for k in ("dice_artery_vein_mean", "cldice_artery_vein_mean",
              "dice_artery", "dice_vein", "dice_overlap",
              "betti0_err_artery", "betti0_err_vein"):
        print(f"  {k:26s} {agg.get(k, float('nan')):.4f}")
    print("\nThe same metric implementation is used by the main training pipeline.")


if __name__ == "__main__":
    main()
