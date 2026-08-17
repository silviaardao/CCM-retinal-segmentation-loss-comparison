"""
Preprocess every AV dataset into one unified format.

All three datasets (DRIVE/RITE, LES-AV, Fundus-AVSeg) use the SAME RGB annotation
scheme (red=artery, blue=vein, green=overlap, white=ambiguous, black=background),
verified pixel-exact with no antialiasing, so a single converter handles all of
them via the existing av_rgb_to_label.

We copy each original image and colour-coded AV mask into a consistent folder
structure without changing their pixel values. During training, the RGB mask
colours are converted in memory to class indices 0–4; this numerical label 
map is not saved over the original mask.

Generated layout:
  data/organized/<dataset>/images/<gid>.<ext>  original fundus image
  data/organized/<dataset>/masks/<gid>.<ext>   original colour AV mask
  data/organized/<dataset>/metadata.csv        per-image metadata and split
  datasets/splits/<split>.csv                  combined cross-dataset manifest

Each globally unique ``gid`` has the form ``<dataset>_<original_id>``.
"""

from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image

from datasets.av_dataset import av_rgb_to_label, extract_image_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
ORGANIZED = PROJECT_ROOT / "data" / "organized"
SPLITS = PROJECT_ROOT / "datasets" / "splits"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".gif")
SPLIT_SEED = 42
VAL_FRACTION = 0.2   # of the train pool, for datasets with an official test split
DISEASE_CODE = {"G": "glaucoma", "N": "normal", "D": "diabetic", "A": "amd"}


# Per-dataset sample iterators
# Each yields dicts: {orig_id, image_path, av_path, disease, official}
# official is "train"/"test"/None (None => we create a 3-way split ourselves).

def _index_by_id(folder):
    out = {}
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() in IMG_EXTS:
            out[extract_image_id(f)] = f
    return out


def iter_drive():
    for split in ("training", "test"):
        img_dir = PROJECT_ROOT / "data" / split / "images"
        av_dir = PROJECT_ROOT / "data" / split / "av"
        if not img_dir.exists() or not av_dir.exists():
            continue
        av = _index_by_id(av_dir)
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            iid = extract_image_id(f)
            if iid in av:
                yield {"orig_id": iid, "image_path": f, "av_path": av[iid],
                       "disease": "", "official": "train" if split == "training" else "test"}


def iter_les_av():
    img_dir = RAW / "les_av" / "images"
    av_dir = RAW / "les_av" / "arteries-and-veins"
    av = _index_by_id(av_dir)
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        iid = extract_image_id(f)
        if iid in av:
            yield {"orig_id": iid, "image_path": f, "av_path": av[iid],
                   "disease": "", "official": None}   # LES-AV has no official split


def _fundus_split_hint():
    root = RAW / "fundus_avseg"
    hint = {}
    for name, tag in (("training.txt", "train"), ("testing.txt", "test")):
        p = root / name
        if p.exists():
            for line in p.read_text().splitlines():
                stem = line.strip()
                if stem:
                    hint[Path(stem).stem] = tag
    return hint


def iter_fundus_avseg():
    img_dir = RAW / "fundus_avseg" / "images"
    av_dir = RAW / "fundus_avseg" / "annotation"
    hint = _fundus_split_hint()
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        stem = f.stem
        av_path = av_dir / f.name
        if not av_path.exists():
            continue
        code = stem.split("_")[-1] if "_" in stem else ""
        yield {"orig_id": stem, "image_path": f, "av_path": av_path,
               "disease": DISEASE_CODE.get(code, ""), "official": hint.get(stem)}


DATASETS = {"drive": iter_drive, "les_av": iter_les_av, "fundus_avseg": iter_fundus_avseg}


# Conversion

def convert_one(sample, images_dir, masks_dir, gid):
    # Copy the ORIGINAL files unchanged so both stay human-viewable.
    img_name = f"{gid}{sample['image_path'].suffix.lower()}"
    mask_name = f"{gid}{sample['av_path'].suffix.lower()}"
    shutil.copy2(sample["image_path"], images_dir / img_name)
    shutil.copy2(sample["av_path"], masks_dir / mask_name)

    # Class stats only (computed in memory, not written to disk).
    label = av_rgb_to_label(np.array(Image.open(sample["av_path"]).convert("RGB")))
    h, w = label.shape
    counts = {c: int((label == c).sum()) for c in range(5)}
    return w, h, counts, img_name, mask_name


def assign_splits(dataset, rows):
    """Fill row['split'] in place. Honours official train/test; carves val out."""
    rng = random.Random(SPLIT_SEED)
    official = [r for r in rows if r["official"] is not None]
    if official:                                          # drive, fundus_avseg
        for r in rows:
            r["split"] = r["official"]
        train_rows = [r for r in rows if r["split"] == "train"]
        rng.shuffle(train_rows)
        n_val = max(1, round(VAL_FRACTION * len(train_rows)))
        for r in train_rows[:n_val]:
            r["split"] = "val"
    else:                                                 # les_av: 3-way split
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n_test = max(1, round(0.20 * len(rows)))
        n_val = max(1, round(0.15 * len(rows)))
        test_ids = set(idx[:n_test])
        val_ids = set(idx[n_test:n_test + n_val])
        for i, r in enumerate(rows):
            r["split"] = "test" if i in test_ids else ("val" if i in val_ids else "train")
