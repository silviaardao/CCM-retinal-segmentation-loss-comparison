"""Small checks for the repository's main models, losses, labels and splits."""

import csv
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.av_dataset import av_rgb_to_label, labels_to_rgb
from evaluation.losses.segmentation_losses import get_multiclass_loss
from models.factory import MODEL_CHOICES, get_model
from scripts.sweep_multiclass import ALL_AXES, build_configs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOSS_NAMES = [
    "ce", "dice", "ce_dice", "weighted_ce_dice", "focal_dice",
    "cf_v", "cf_b", "cf_vb", "cbav", "cldice",
]


class CoreChecks(unittest.TestCase):
    def test_models_return_five_class_logits(self):
        image = torch.randn(1, 3, 64, 64)
        for name in MODEL_CHOICES:
            with self.subTest(model=name):
                model = get_model(name, in_channels=3, out_channels=5, features_start=8)
                model.eval()
                with torch.no_grad():
                    output = model(image)
                self.assertEqual(tuple(output.shape), (1, 5, 64, 64))

    def test_losses_are_finite_and_support_backpropagation(self):
        target = torch.randint(0, 5, (1, 64, 64))
        for name in LOSS_NAMES:
            with self.subTest(loss=name):
                logits = torch.randn(1, 5, 64, 64, requires_grad=True)
                value = get_multiclass_loss(name, num_classes=5)(logits, target)
                self.assertEqual(value.ndim, 0)
                self.assertTrue(torch.isfinite(value).item())
                value.backward()
                self.assertIsNotNone(logits.grad)
                self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_label_colour_round_trip(self):
        labels = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        converted = av_rgb_to_label(labels_to_rgb(labels))
        np.testing.assert_array_equal(converted, labels)

    def test_split_ids_do_not_overlap(self):
        seen = set()
        for split in ("train", "val", "test"):
            path = PROJECT_ROOT / "datasets" / "splits" / f"{split}.csv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            ids = {row["gid"] for row in rows}
            self.assertEqual(len(ids), len(rows))
            self.assertTrue(seen.isdisjoint(ids))
            self.assertTrue(all(row["split"] == split for row in rows))
            seen.update(ids)

    def test_study_sweep_contains_recorded_ninety_runs(self):
        configs = build_configs(ALL_AXES, "study")
        self.assertEqual(len(configs), 90)
        seed_42 = {(row["model"], row["loss"]) for row in configs if row["seed"] == 42}
        self.assertEqual(len(seed_42), 50)
        for seed in (123, 456, 7, 2024):
            rows = [row for row in configs if row["seed"] == seed]
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row["model"] == "basic_unet" for row in rows))

    def test_committed_nnunet_folds_cover_the_training_pool(self):
        path = PROJECT_ROOT / "nnunet" / "splits_final.json"
        folds = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(folds), 5)
        for fold in folds:
            train, val = set(fold["train"]), set(fold["val"])
            self.assertTrue(train.isdisjoint(val))
            self.assertEqual(len(train | val), 118)

    def test_compact_result_matrix_has_every_factorial_cell(self):
        path = PROJECT_ROOT / "results" / "summary" / "architecture_loss_matrix.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        model_rows = [row for row in rows if row["model"] != "loss_mean"]
        self.assertEqual(len(model_rows), 5)
        for row in model_rows:
            self.assertTrue(all(row[name] for name in LOSS_NAMES))


if __name__ == "__main__":
    unittest.main()
