"""
Convert the organized AV data into nnU-Net v2 Dataset format for benchmarking.

- imagesTr/labelsTr = our TRAIN + VAL (nnU-Net runs its own internal 5-fold CV).
- imagesTs (+ labelsTs, kept for OUR evaluation) = our held-out TEST set, so the
  comparison is on the exact same test images our models were scored on.

RGB fundus -> 3 channel files (_0000=R, _0001=G, _0002=B).
Colour AV mask -> single-channel integer label 0..4 via av_rgb_to_label.

Output: <nnUNet_raw>/Dataset<ID>_<NAME>/{imagesTr,labelsTr,imagesTs,labelsTs}/ + dataset.json
"""

from pathlib import Path
import sys
import argparse
import json

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from datasets.av_dataset import av_rgb_to_label
from datasets.manifest_dataset import load_manifest

LABELS = {"background": 0, "artery": 1, "vein": 2, "overlap": 3, "ambiguous": 4}


def save_channels(image_path, out_dir, gid):
    rgb = Image.open(image_path).convert("RGB")
    for ch in range(3):
        band = rgb.getchannel(ch)                 # R, G, B
        band.save(out_dir / f"{gid}_{ch:04d}.png")


def save_label(mask_path, out_dir, gid):
    label = av_rgb_to_label(np.array(Image.open(mask_path).convert("RGB")))
    Image.fromarray(label.astype(np.uint8), mode="L").save(out_dir / f"{gid}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=1)
    ap.add_argument("--name", type=str, default="RetinaAV")
    ap.add_argument("--datasets", nargs="+", default=["drive", "les_av", "fundus_avseg"])
    ap.add_argument("--raw-root", type=str, default=str(PROJECT_ROOT / "nnunet" / "nnUNet_raw"))
    args = ap.parse_args()

    ds_dir = Path(args.raw_root) / f"Dataset{args.dataset_id:03d}_{args.name}"
    imagesTr, labelsTr = ds_dir / "imagesTr", ds_dir / "labelsTr"
    imagesTs, labelsTs = ds_dir / "imagesTs", ds_dir / "labelsTs"
    for d in (imagesTr, labelsTr, imagesTs, labelsTs):
        d.mkdir(parents=True, exist_ok=True)

    # nnU-Net does its own CV, so train + val both go into the training pool.
    train_samples = load_manifest("train", args.datasets) + load_manifest("val", args.datasets)
    test_samples = load_manifest("test", args.datasets)

    print(f"Training pool: {len(train_samples)} | Test: {len(test_samples)}")
    for gid, img, mask in train_samples:
        save_channels(img, imagesTr, gid)
        save_label(mask, labelsTr, gid)
    for gid, img, mask in test_samples:
        save_channels(img, imagesTs, gid)
        save_label(mask, labelsTs, gid)        # kept for OUR eval; nnU-Net ignores it

    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": LABELS,
        "numTraining": len(train_samples),
        "file_ending": ".png",
    }
    with open(ds_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Wrote nnU-Net dataset -> {ds_dir}")
    print(f"  imagesTr: {len(list(imagesTr.glob('*.png')))} files "
          f"({len(train_samples)} cases x 3 channels)")
    print(f"  labelsTr: {len(list(labelsTr.glob('*.png')))}")
    print(f"  imagesTs: {len(list(imagesTs.glob('*.png')))} files ({len(test_samples)} cases)")


if __name__ == "__main__":
    main()
