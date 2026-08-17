"""
Manifest-driven dataset: reads samples from a unified split CSV (datasets/splits/*.csv)
rather than fixed per-dataset folders, so training can mix DRIVE, LES-AV and
Fundus-AVSeg freely.

Masks on disk are the ORIGINAL colour AV masks (human-viewable); the RGB->5-class
index conversion happens here in memory via av_rgb_to_label. Augmentation and
resizing are inherited from AVDataset.
(geometric transforms apply to both image and mask; photometric to the image only).
"""

from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.av_dataset import AVDataset, av_rgb_to_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_manifest(split, datasets=None):
    """Read datasets/splits/<split>.csv, optionally filtered to given datasets.

    Returns a list of (gid, abs_image_path, abs_mask_path).
    """
    path = PROJECT_ROOT / "datasets" / "splits" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}\n"
                                "Run: python scripts/preprocess_datasets.py --dataset all")
    keep = set(datasets) if datasets else None
    samples = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if keep is not None and row["dataset"] not in keep:
                continue
            samples.append((row["gid"],
                            PROJECT_ROOT / row["image_path"],
                            PROJECT_ROOT / row["mask_path"]))
    return samples


class ManifestDataset(AVDataset):
    """Multiclass AV dataset backed by an explicit sample list (no folder scan)."""

    def __init__(self, samples, image_size=512, augment=False, input_channels=3):
        Dataset.__init__(self)
        self.samples = samples                 # list of (gid, image_path, mask_path)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.input_channels = int(input_channels)
        self.task = "multiclass"
        self.return_cbav_masks = False
        if not self.samples:
            raise RuntimeError("ManifestDataset received an empty sample list.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("L" if self.input_channels == 1 else "RGB")
        mask = Image.open(mask_path).convert("RGB")          # colour AV mask

        image, mask = self._resize_pair(image, mask)         # inherited
        if self.augment:
            image, mask = self._augment_pair(image, mask)    # inherited

        image_np = np.array(image).astype(np.float32) / 255.0
        if self.input_channels == 1:
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        label = av_rgb_to_label(np.array(mask))              # RGB -> 0..4 in memory
        return image_tensor, torch.from_numpy(label).long()
