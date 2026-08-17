"""
Score every loss in the nnU-Net objective experiment with the study's metrics.

Reads each predictions folder under nnunet/predictions_cf/<loss>/ (label PNGs 0..4),
scores against the same held-out test GT at 512 (identical to evaluate_nnunet.py),
and writes one tidy summary row per loss.

    py -3.14 scripts/score_cf_experiment.py
"""

from pathlib import Path
import sys
import csv
import json

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
EVAL_SIZE = 512
PRED_ROOT = PROJECT_ROOT / "nnunet" / "predictions_cf"
GT_DIR = PROJECT_ROOT / "nnunet" / "nnUNet_raw" / "Dataset001_RetinaAV" / "labelsTs"
OUT_DIR = PROJECT_ROOT / "results" / "nnunet_cf_experiment"


def to_eval(arr):
    return np.array(Image.fromarray(arr.astype(np.uint8)).resize((EVAL_SIZE, EVAL_SIZE), Image.NEAREST))


def as_logits(label_map):
    t = torch.from_numpy(label_map.astype(np.int64)).unsqueeze(0)
    return F.one_hot(t, NUM_CLASSES).permute(0, 3, 1, 2).float() * 10.0


def expected_ids():
    ids = {p.stem for p in GT_DIR.glob("*.png")}
    if len(ids) != 44:
        raise RuntimeError(f"Expected 44 test labels in {GT_DIR}, found {len(ids)}.")
    return ids


def score_dir(pred_dir):
    expected = expected_ids()
    predicted = {p.stem for p in pred_dir.glob("*.png")}
    missing = sorted(expected - predicted)
    unexpected = sorted(predicted - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {len(missing)}: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {len(unexpected)}: {', '.join(unexpected)}")
        raise RuntimeError(f"Incomplete prediction set in {pred_dir}: " + "; ".join(details))

    records = []
    for gid in sorted(expected):
        p = pred_dir / f"{gid}.png"
        gt_p = GT_DIR / p.name
        pred = to_eval(np.array(Image.open(p)))
        gt = to_eval(np.array(Image.open(gt_p)))
        target = torch.from_numpy(gt.astype(np.int64)).unsqueeze(0)
        rec = {"gid": gid,
               **per_image_multiclass_dice(as_logits(pred), target, num_classes=NUM_CLASSES)[0],
               **per_image_topology(as_logits(pred), target)[0]}
        records.append(rec)
    return records


def main():
    if not PRED_ROOT.exists():
        raise SystemExit(f"No predictions at {PRED_ROOT} yet.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for pred_dir in sorted(PRED_ROOT.iterdir()):
        if not pred_dir.is_dir():
            continue
        recs = score_dir(pred_dir)
        if not recs:
            print(f"  {pred_dir.name}: no scored images (skipped)")
            continue
        metric_records = [{k: v for k, v in rec.items() if k != "gid"} for rec in recs]
        agg = aggregate_dice_scores(metric_records)
        rows.append({"loss": pred_dir.name, "n": len(recs), **agg})
        # Keep per-image scores so aggregate results can be checked if needed.
        with open(OUT_DIR / f"per_image_{pred_dir.name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader(); w.writerows(recs)
        print(f"  {pred_dir.name:24s} AV Dice {agg.get('dice_artery_vein_mean', float('nan')):.4f}  "
              f"overlap {agg.get('dice_overlap', float('nan')):.4f}")

    if not rows:
        raise SystemExit("Nothing scored yet.")
    keys = ["loss", "n"] + [k for k in rows[0] if k not in ("loss", "n")]
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    rows.sort(key=lambda r: r.get("dice_artery_vein_mean", 0), reverse=True)
    print("\nAV Dice by objective (nnU-Net pipeline, 100 epochs, fold 0)")
    for r in rows:
        print(f"  {r['loss']:24s} {r.get('dice_artery_vein_mean', float('nan')):.4f}")
    print(f"\nWrote {OUT_DIR/'summary.csv'}")


if __name__ == "__main__":
    main()
