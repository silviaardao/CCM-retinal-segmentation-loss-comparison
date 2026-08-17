"""
Preprocess AV datasets into the unified 5-class format and write split manifests.

Usage:
  python scripts/preprocess_datasets.py --dataset all
  python scripts/preprocess_datasets.py --dataset les_av fundus_avseg
"""

from pathlib import Path
import sys
import argparse
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import preprocessing as P

META_FIELDS = ["gid", "dataset", "orig_id", "split", "disease", "width", "height",
               "image_path", "mask_path",
               "px_bg", "px_artery", "px_vein", "px_overlap", "px_ambiguous"]
SPLIT_FIELDS = ["gid", "dataset", "split", "image_path", "mask_path"]


def rel(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def process_dataset(dataset):
    out_dir = P.ORGANIZED / dataset
    images_dir, masks_dir = out_dir / "images", out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    samples = list(P.DATASETS[dataset]())
    if not samples:
        print(f"[{dataset}] no samples found — skipping")
        return []
    print(f"[{dataset}] organizing {len(samples)} images (copied unchanged)...")

    rows = []
    for s in samples:
        gid = f"{dataset}_{s['orig_id']}"
        w, h, counts, img_name, mask_name = P.convert_one(s, images_dir, masks_dir, gid)
        rows.append({
            "gid": gid, "dataset": dataset, "orig_id": s["orig_id"],
            "official": s["official"], "disease": s["disease"],
            "width": w, "height": h,
            "image_path": rel(images_dir / img_name),
            "mask_path": rel(masks_dir / mask_name),
            "px_bg": counts[0], "px_artery": counts[1], "px_vein": counts[2],
            "px_overlap": counts[3], "px_ambiguous": counts[4],
        })

    P.assign_splits(dataset, rows)

    with open(out_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    print(f"[{dataset}] done. split -> train {counts['train']}, "
          f"val {counts['val']}, test {counts['test']}")
    return rows


def write_split_manifests(all_rows):
    SPLITS = P.SPLITS
    SPLITS.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        subset = [r for r in all_rows if r["split"] == split]
        with open(SPLITS / f"{split}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SPLIT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)
    print(f"\nSplit manifests written to {rel(SPLITS)}/ "
          f"(train {sum(r['split']=='train' for r in all_rows)}, "
          f"val {sum(r['split']=='val' for r in all_rows)}, "
          f"test {sum(r['split']=='test' for r in all_rows)})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="+", default=["all"],
                   choices=["all"] + list(P.DATASETS.keys()))
    return p.parse_args()


def main():
    args = parse_args()
    datasets = list(P.DATASETS.keys()) if "all" in args.dataset else args.dataset
    all_rows = []
    for d in datasets:
        all_rows.extend(process_dataset(d))
    if all_rows:
        write_split_manifests(all_rows)


if __name__ == "__main__":
    main()
