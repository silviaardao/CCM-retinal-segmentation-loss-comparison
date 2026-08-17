"""Shared dataset utilities for five-class retinal artery-vein segmentation.

5-class scheme:
  0 = background  (black)
  1 = artery      (red)
  2 = vein        (blue)
  3 = overlap     (green) — artery crossing over vein
  4 = ambiguous   (white) — annotator couldn't determine type
"""

from pathlib import Path
import re
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageFilter


NUM_CLASSES = 5

CLASS_NAMES = {
    0: "background",
    1: "artery",
    2: "vein",
    3: "overlap",
    4: "ambiguous",
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (0, 255, 0),
    4: (255, 255, 255),
}


def extract_image_id(path: Path) -> str:
    match = re.search(r"\d+", path.stem)
    if match is None:
        return path.stem
    return match.group(0)


def av_rgb_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB AV mask to 5-class label map.

    Any unexpected colour maps to ambiguous (4) to avoid silent data loss.
    """
    if mask_rgb.ndim != 3 or mask_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB mask [H,W,3], got {mask_rgb.shape}")

    r, g, b = mask_rgb[:, :, 0], mask_rgb[:, :, 1], mask_rgb[:, :, 2]

    # Default to ambiguous — catches antialiased edges and unexpected colours
    label = np.full(mask_rgb.shape[:2], fill_value=4, dtype=np.int64)

    label[(r == 0) & (g == 0) & (b == 0)] = 0       # background
    label[(r == 255) & (g == 0) & (b == 0)] = 1      # artery
    label[(r == 0) & (g == 0) & (b == 255)] = 2      # vein
    label[(r == 0) & (g == 255) & (b == 0)] = 3      # overlap
    label[(r == 255) & (g == 255) & (b == 255)] = 4   # ambiguous

    return label


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[labels == class_id] = color
    return rgb


class AVDataset(Dataset):
    """
    Folder-based retinal AV dataset loader.

    task:
      "binary"     -> vessel/background from vessel/ folder
      "multiclass" -> 5-class AV from av/ folder
    """

    def __init__(
        self,
        root_dir,
        split="training",
        task="binary",
        image_size=512,
        augment=False,
        input_channels=3,
        return_cbav_masks=False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.task = task
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.input_channels = int(input_channels)
        self.return_cbav_masks = bool(return_cbav_masks)

        if self.task not in ["binary", "multiclass"]:
            raise ValueError("task must be 'binary' or 'multiclass'")
        if self.input_channels not in [1, 3]:
            raise ValueError("input_channels must be 1 or 3")

        self.image_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / ("vessel" if self.task == "binary" else "av")
        self.cbav_dir = self.root_dir / split / "cbav_masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask folder not found: {self.mask_dir}")

        image_files = sorted(
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        )
        mask_files = sorted(
            f for f in self.mask_dir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        )
        mask_by_id = {extract_image_id(f): f for f in mask_files}

        self.samples = []
        for image_file in image_files:
            image_id = extract_image_id(image_file)
            if image_id in mask_by_id:
                self.samples.append((image_id, image_file, mask_by_id[image_id]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image/mask pairs for split={split}, task={task}."
            )

        if self.return_cbav_masks and not self.cbav_dir.exists():
            raise FileNotFoundError(
                f"Precomputed CBAV masks were requested but not found: {self.cbav_dir}. "
                "The published multiclass pipeline derives its structure masks from "
                "each target at runtime and does not require this option."
            )

    def __len__(self):
        return len(self.samples)

    def get_sample_paths(self, idx):
        _, image_path, mask_path = self.samples[idx]
        return image_path, mask_path

    def _resize_pair(self, image, mask):
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        return image, mask

    # Augmentation

    def _augment_pair(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = image.rotate(90 * k, resample=Image.BILINEAR)
            mask = mask.rotate(90 * k, resample=Image.NEAREST)

        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            image = self._gamma_correction(image, gamma=random.uniform(0.7, 1.5))
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.3:
            image = self._add_gaussian_noise(image, std=random.uniform(0.01, 0.05))

        return image, mask

    @staticmethod
    def _gamma_correction(image, gamma):
        arr = np.array(image).astype(np.float32) / 255.0
        arr = np.clip(np.power(arr, gamma), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8), mode=image.mode)

    @staticmethod
    def _add_gaussian_noise(image, std=0.03):
        arr = np.array(image).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.normal(0, std, arr.shape).astype(np.float32), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8), mode=image.mode)

    # CBAV masks

    def _load_cbav_mask(self, image_id: str, name: str) -> torch.Tensor:
        path = self.cbav_dir / f"{image_id}_{name}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing CBAV mask: {path}")
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        arr = (np.array(mask) > 127).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0).float()

    # Sample loading

    def __getitem__(self, idx):
        image_id, image_path, mask_path = self.samples[idx]

        image = Image.open(image_path).convert("L" if self.input_channels == 1 else "RGB")
        mask = Image.open(mask_path).convert("L" if self.task == "binary" else "RGB")

        image, mask = self._resize_pair(image, mask)
        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image_np = np.array(image).astype(np.float32) / 255.0
        if self.input_channels == 1:
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        if self.task == "binary":
            mask_np = (np.array(mask) > 127).astype(np.float32)
            return image_tensor, torch.from_numpy(mask_np).unsqueeze(0).float()

        mask_np = av_rgb_to_label(np.array(mask))
        mask_tensor = torch.from_numpy(mask_np).long()

        if not self.return_cbav_masks:
            return image_tensor, mask_tensor

        cbav = {
            "cross": self._load_cbav_mask(image_id, "cross"),
            "branch_artery": self._load_cbav_mask(image_id, "branch_artery"),
            "branch_vein": self._load_cbav_mask(image_id, "branch_vein"),
        }
        return image_tensor, mask_tensor, cbav
