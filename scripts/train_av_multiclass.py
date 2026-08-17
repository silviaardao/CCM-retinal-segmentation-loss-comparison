"""
Five-class retinal artery-vein segmentation training.

Classes: 0=background, 1=artery, 2=vein, 3=overlap, 4=ambiguous

Study training configuration:
  - Adam beta1=0.5, beta2=0.999
  - lr=4e-4, ReduceLROnPlateau (halve after 50 epochs plateau)
  - Early stopping after 50 epochs with no validation AV Dice improvement
  - Image size 512

The Adam beta values and scheduler form follow the CF-Loss training recipe. The
learning rate and early-stopping budget above are the values recorded in this
study's completed main-run configurations.

Class weights used by the weighted losses: [1.0, 3.0, 3.0, 5.0, 2.0].
The larger vessel-class weights reduce the influence of the dominant background
class; the overlap class receives the largest weight because it is rare.
"""

from pathlib import Path
import sys
import argparse
import random
import json
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.av_dataset import labels_to_rgb
from datasets.manifest_dataset import ManifestDataset, load_manifest
from models.factory import get_model, MODEL_CHOICES

ALL_DATASETS = ["drive", "les_av", "fundus_avseg"]
from evaluation.losses.segmentation_losses import get_multiclass_loss
from evaluation.metrics.segmentation_metrics import (
    aggregate_dice_scores,
    confusion_matrix_from_logits,
    pixel_accuracy_from_confusion_matrix,
    vessel_density_errors_from_logits,
    per_image_multiclass_dice,
    per_image_topology,
)

NUM_CLASSES = 5
CLASS_NAMES = {0: "background", 1: "artery", 2: "vein", 3: "overlap", 4: "ambiguous"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_prediction_preview(model, dataloader, device, output_path):
    model.eval()
    images, masks = next(iter(dataloader))
    images = images.to(device)
    with torch.no_grad():
        preds = torch.argmax(model(images), dim=1)

    image = images[0].detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)
    gt_rgb = labels_to_rgb(masks[0].detach().cpu().numpy())
    pred_rgb = labels_to_rgb(preds[0].detach().cpu().numpy())

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(image); plt.title("Input image"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(gt_rgb); plt.title("Ground truth"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(pred_rgb); plt.title("Prediction"); plt.axis("off")
    plt.tight_layout(); plt.savefig(output_path, dpi=200); plt.close()


def save_confusion_matrix_plot(cm, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar()
    ticks = list(CLASS_NAMES.keys())
    labels = [CLASS_NAMES[i] for i in ticks]
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.tight_layout(); plt.savefig(output_path, dpi=200); plt.close()


def evaluate(model, dataloader, criterion, device, topology=False):
    """
    Evaluate over a whole dataloader. Dice is aggregated per-image and
    presence-conditioned (see metrics module), so the result is independent of
    batch size. Density metrics are image-count weighted for the same reason.
    topology=True additionally computes clDice + Betti-0 (connectivity)
    — CPU-heavy skeletonisation, so pass it only for the final test evaluation.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_images = 0
    all_per_image = []
    all_topology = []
    density_sum = {}
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            total_batches += 1
            b = images.shape[0]
            total_images += b

            all_per_image.extend(per_image_multiclass_dice(logits, masks, num_classes=NUM_CLASSES))
            if topology:
                all_topology.extend(per_image_topology(logits, masks))
            for k, v in vessel_density_errors_from_logits(logits, masks, num_classes=NUM_CLASSES).items():
                density_sum[k] = density_sum.get(k, 0.0) + v * b
            total_cm += confusion_matrix_from_logits(logits, masks, num_classes=NUM_CLASSES)

    mean_scores = aggregate_dice_scores(all_per_image)
    if all_topology:
        mean_scores.update(aggregate_dice_scores(all_topology))   # nanmean of clDice/Betti-0
    for k, v in density_sum.items():
        mean_scores[k] = v / max(total_images, 1)
    mean_scores["pixel_accuracy"] = pixel_accuracy_from_confusion_matrix(total_cm)
    return total_loss / max(total_batches, 1), mean_scores, total_cm, all_per_image


def save_history_csv(history, path):
    if not history:
        return
    keys = sorted(set().union(*(row.keys() for row in history)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_per_image_csv(per_image_scores, samples, path):
    if not per_image_scores:
        return
    keys = sorted(per_image_scores[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_idx", "gid", "dataset", "image_path"] + keys
        )
        writer.writeheader()
        for i, scores in enumerate(per_image_scores):
            gid, image_path, _ = samples[i]
            dataset = next((name for name in ALL_DATASETS if gid.startswith(f"{name}_")), "")
            try:
                image_path = image_path.relative_to(PROJECT_ROOT)
            except ValueError:
                pass
            writer.writerow({
                "image_idx": i,
                "gid": gid,
                "dataset": dataset,
                "image_path": str(image_path).replace("\\", "/"),
                **scores,
            })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="basic_unet", choices=MODEL_CHOICES)
    p.add_argument("--loss", type=str, default="weighted_ce_dice",
                   choices=["ce", "dice", "ce_dice", "weighted_ce_dice", "focal_dice",
                            "cf_v", "cf_b", "cf_vb", "cbav", "cldice"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--features-start", type=int, default=32)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                   choices=ALL_DATASETS,
                   help="Which datasets to train/select/report on. Splits are read "
                        "from datasets/splits/{train,val,test}.csv (train/val/test are "
                        "fixed there by preprocess_datasets.py).")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    # Note: cbav is a normal loss here — CBAVLoss derives its crossover/branch
    # structure from the target on the fly (see evaluation/losses/structure_masks.py), so it
    # goes through the same criterion(logits, masks) path as every other loss.

    run_name = args.run_name or f"{args.model}_{args.loss}_5class"
    out_dir = PROJECT_ROOT / "results" / "multiclass" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir:", out_dir)

    # Train/val/test are read from the unified split manifests (fixed by
    # preprocess_datasets.py). Selection/early-stopping use val; test is held out
    # and only evaluated once, at the end. Train is augmented; val/test are not.
    train_ds = ManifestDataset(load_manifest("train", args.datasets),
                               image_size=args.image_size, augment=True, input_channels=3)
    val_ds = ManifestDataset(load_manifest("val", args.datasets),
                             image_size=args.image_size, augment=False, input_channels=3)
    test_ds = ManifestDataset(load_manifest("test", args.datasets),
                              image_size=args.image_size, augment=False, input_channels=3)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Datasets: {args.datasets} | Train: {len(train_ds)}, "
          f"Val: {len(val_ds)}, Test: {len(test_ds)}")

    model = get_model(args.model, in_channels=3, out_channels=NUM_CLASSES,
                      features_start=args.features_start).to(device)

    # Class weights: bg=1, artery/vein=3, overlap=5, ambiguous=2
    # Ambiguous at 2.0 not 6.0: the model shouldn't be heavily rewarded
    # for predicting a class whose ground truth is inherently uncertain.
    weights_list = [1.0, 3.0, 3.0, 5.0, 2.0]
    class_weights = torch.tensor(weights_list, dtype=torch.float32, device=device)
    criterion = get_multiclass_loss(
        args.loss, num_classes=NUM_CLASSES, class_weights=class_weights
    )

    # Adam with beta1=0.5 per CF-Loss paper
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )

    config = vars(args).copy()
    config.update({
        "device": str(device), "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES, "class_weights": weights_list,
        "adam_betas": [0.5, 0.999],
        "datasets": args.datasets,
        "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds),
        "selection_metric": "val dice_artery_vein_mean",
    })
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    best_av_dice = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            train_batches += 1

        train_loss = train_loss_total / max(train_batches, 1)
        val_loss, val_scores, val_cm, val_per_image = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "lr": current_lr, **val_scores}
        history.append(row)
        av_dice = val_scores["dice_artery_vein_mean"]
        if np.isnan(av_dice):
            av_dice = -1.0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | lr {current_lr:.2e} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"val AV Dice {av_dice:.4f} | artery {val_scores['dice_artery']:.4f} | "
            f"vein {val_scores['dice_vein']:.4f} | "
            f"overlap {val_scores['dice_overlap']:.4f} | "
            f"ambiguous {val_scores['dice_ambiguous']:.4f}"
        )

        # Model selection on VALIDATION only.
        if av_dice > best_av_dice:
            best_av_dice = av_dice
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            with open(out_dir / "best_val_metrics.json", "w") as f:
                json.dump(row, f, indent=4)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
            break

    torch.save(model.state_dict(), out_dir / "final_model.pth")
    save_history_csv(history, out_dir / "training_history.csv")

    # Load the checkpoint selected on validation data, then evaluate it once on
    # the held-out test set. Configuration selection uses validation results.
    best_ckpt = out_dir / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_scores, test_cm, test_per_image = evaluate(
        model, test_loader, criterion, device, topology=True
    )
    test_row = {"test_loss": test_loss, "best_val_av_dice": best_av_dice, **test_scores}
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_row, f, indent=4)
    save_confusion_matrix_plot(test_cm, out_dir / "test_confusion_matrix.png")
    save_prediction_preview(model, test_loader, device, out_dir / "test_prediction_preview.png")
    save_per_image_csv(test_per_image, test_ds.samples, out_dir / "test_per_image_dice.csv")

    print(f"\nTraining complete. Best VAL AV Dice: {best_av_dice:.4f} | "
          f"TEST AV Dice: {test_scores['dice_artery_vein_mean']:.4f}")
    print("Results saved to:", out_dir)


if __name__ == "__main__":
    main()
