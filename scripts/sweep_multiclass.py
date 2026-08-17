"""
Sweep driver for train_av_multiclass.py.

Constructs the architecture, loss and seed configurations used by the controlled
study. Study mode reproduces the recorded 90-run design: every architecture and
loss at seed 42, followed by the other four seeds for every Basic U-Net loss.
Coordinate and grid modes remain available for optional sensitivity experiments.

Each config is launched through the training script's command-line interface.
Runs are written under results/multiclass/. The sweep is resumable: completed
runs with test_metrics.json are skipped unless --force is supplied.

Usage:
  python scripts/sweep_multiclass.py --dry-run
  python scripts/sweep_multiclass.py --axes lr --epochs 2 --patience 2
  python scripts/sweep_multiclass.py
"""

from pathlib import Path
import sys
import argparse
import json
import csv
import itertools
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_av_multiclass.py"
RESULTS_DIR = PROJECT_ROOT / "results" / "multiclass"

# Shared reference configuration for factors not selected in the sweep.
BASELINE = {
    "model": "basic_unet",
    "loss": "ce_dice",
    "lr": 4e-4,
    "batch_size": 2,
    "features_start": 32,
    "image_size": 512,
    "seed": 42,
}

# Values tried per axis (baseline value included; deduped later).
AXIS_VALUES = {
    "model": ["basic_unet", "residual_unet", "attention_unet", "unetpp", "resunetpp"],
    "loss": ["ce", "dice", "ce_dice", "weighted_ce_dice", "focal_dice",
             "cf_v", "cf_b", "cf_vb", "cbav", "cldice"],
    "lr": [1e-4, 4e-4, 8e-4, 2e-3],
    "batch_size": [2, 4, 8],
    "features_start": [16, 32, 64],
    "seed": [42, 123, 456, 7, 2024],
}

ALL_AXES = list(AXIS_VALUES.keys())


ALL_DS = ["drive", "les_av", "fundus_avseg"]


def ds_tag(datasets):
    """Short, self-documenting tag for which datasets a run used."""
    datasets = list(datasets) if datasets else ALL_DS
    if set(datasets) == set(ALL_DS):
        return "all"
    return "-".join(sorted(datasets))


def run_name_for(cfg):
    # Seed suffix only for non-default seeds, so the existing seed-42 runs keep
    # their names (and get reused) while extra seeds are disambiguated.
    seed_suffix = "" if cfg.get("seed", 42) == 42 else f"_s{cfg['seed']}"
    return (
        f"sweep_{ds_tag(cfg.get('datasets'))}"
        f"_{cfg['model']}"
        f"_loss-{cfg['loss']}"
        f"_lr{cfg['lr']:.1e}"
        f"_bs{cfg['batch_size']}"
        f"_fs{cfg['features_start']}"
        + seed_suffix
    )


def build_configs(axes, mode):
    """
    mode="study":      recorded model/loss comparison and Basic U-Net multi-seed runs.
    mode="coordinate": baseline plus one-axis-at-a-time variations (sensitivity).
    mode="grid":       full factorial over the chosen axes (every combination) —
                       lets each loss be tuned fairly for a best-vs-best ranking.
    Axes not listed are held at their baseline value.
    """
    configs = []
    seen = set()

    def add(cfg):
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    if mode == "study":
        for model in AXIS_VALUES["model"]:
            for loss in AXIS_VALUES["loss"]:
                cfg = dict(BASELINE)
                cfg.update({"model": model, "loss": loss, "seed": 42})
                add(cfg)
        for seed in AXIS_VALUES["seed"]:
            if seed == 42:
                continue
            for loss in AXIS_VALUES["loss"]:
                cfg = dict(BASELINE)
                cfg.update({"model": "basic_unet", "loss": loss, "seed": seed})
                add(cfg)
    elif mode == "coordinate":
        add(dict(BASELINE))  # anchor point for every axis
        for axis in axes:
            for value in AXIS_VALUES[axis]:
                cfg = dict(BASELINE)
                cfg[axis] = value
                add(cfg)
    elif mode == "grid":
        value_lists = [AXIS_VALUES[a] for a in axes]
        for combo in itertools.product(*value_lists):
            cfg = dict(BASELINE)
            for axis, value in zip(axes, combo):
                cfg[axis] = value
            add(cfg)
    else:
        raise SystemExit(f"Unknown mode: {mode}")
    return configs


def build_command(cfg, epochs, patience):
    run_name = run_name_for(cfg)
    datasets = list(cfg.get("datasets") or ALL_DS)
    return [
        sys.executable, str(TRAIN_SCRIPT),
        "--model", str(cfg["model"]),
        "--loss", str(cfg["loss"]),
        "--lr", repr(cfg["lr"]),
        "--batch-size", str(cfg["batch_size"]),
        "--features-start", str(cfg["features_start"]),
        "--image-size", str(cfg["image_size"]),
        "--seed", str(cfg["seed"]),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--datasets", *datasets,
        "--run-name", run_name,
    ], run_name


def run_one(cfg, epochs, patience, force):
    cmd, run_name = build_command(cfg, epochs, patience)
    out_dir = RESULTS_DIR / run_name
    if not force and (out_dir / "test_metrics.json").exists():
        print(f"[skip] {run_name} (test_metrics.json exists)")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.txt"
    print(f"[run ] {run_name}")
    print("       " + " ".join(cmd))

    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"[FAIL] {run_name} exited with code {proc.returncode}")
        return False
    print(f"[done] {run_name}")
    return True


def _load_json(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _as_float(x):
    return x if isinstance(x, (int, float)) else -1.0


def write_sweep_summary(configs):
    """
    Sort completed runs by validation AV Dice and show their test metrics.

    Validation results are used to select configurations. Test results are kept
    as held-out measurements and are not used for configuration selection.
    """
    rows = []
    for cfg in configs:
        run_name = run_name_for(cfg)
        run_dir = RESULTS_DIR / run_name
        val = _load_json(run_dir / "best_val_metrics.json")
        test = _load_json(run_dir / "test_metrics.json")
        if not test:  # run did not finish
            continue
        rows.append({
            "run_name": run_name,
            "model": cfg["model"],
            "loss": cfg["loss"],
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "features_start": cfg["features_start"],
            "val_av_dice": val.get("dice_artery_vein_mean", ""),
            "test_av_dice": test.get("dice_artery_vein_mean", ""),
            "test_cldice_av": test.get("cldice_artery_vein_mean", ""),
            "test_betti0_artery": test.get("betti0_err_artery", ""),
            "test_betti0_vein": test.get("betti0_err_vein", ""),
            "test_dice_artery": test.get("dice_artery", ""),
            "test_dice_vein": test.get("dice_vein", ""),
            "test_dice_overlap": test.get("dice_overlap", ""),
            "test_dice_ambiguous": test.get("dice_ambiguous", ""),
            "test_pixel_accuracy": test.get("pixel_accuracy", ""),
            "best_epoch": val.get("epoch", ""),
        })

    if not rows:
        print("No completed runs to summarize.")
        return

    rows.sort(key=lambda r: _as_float(r["val_av_dice"]), reverse=True)
    out_path = RESULTS_DIR / "sweep_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSweep summary saved to:", out_path)
    print(f"{'run_name':50s} {'val AV':>8s} {'test AV':>8s}")
    for r in rows:
        v = f"{r['val_av_dice']:.4f}" if isinstance(r["val_av_dice"], (int, float)) else str(r["val_av_dice"])
        t = f"{r['test_av_dice']:.4f}" if isinstance(r["test_av_dice"], (int, float)) else str(r["test_av_dice"])
        print(f"{r['run_name']:50s} {v:>8s} {t:>8s}")
    selected = rows[0]
    print(f"\nSelected by validation AV Dice: {selected['run_name']} "
          f"| validation {selected['val_av_dice']} | test {selected['test_av_dice']}")

    write_best_per_loss(rows)


def write_best_per_loss(rows):
    """
    For each model and loss, select the configuration with the highest validation
    AV Dice and record the corresponding held-out test metrics.
    """
    best_by_key = {}
    for r in rows:
        key = (r.get("model", ""), r["loss"])
        if key not in best_by_key or _as_float(r["val_av_dice"]) > _as_float(best_by_key[key]["val_av_dice"]):
            best_by_key[key] = r

    ranked = sorted(best_by_key.values(), key=lambda r: _as_float(r["test_av_dice"]), reverse=True)
    out_path = RESULTS_DIR / "best_per_loss.csv"
    fields = ["model", "loss", "run_name", "lr", "batch_size", "features_start",
              "val_av_dice", "test_av_dice", "test_cldice_av",
              "test_betti0_artery", "test_betti0_vein",
              "test_dice_artery", "test_dice_vein",
              "test_dice_overlap", "test_pixel_accuracy", "best_epoch"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)

    print("\nBest per (model, loss) - tuned on val, reported on test -", out_path)
    print(f"{'model':16s} {'loss':18s} {'best config':28s} {'test AV':>8s}")
    for r in ranked:
        t = f"{r['test_av_dice']:.4f}" if isinstance(r["test_av_dice"], (int, float)) else str(r["test_av_dice"])
        cfg = f"lr{r['lr']:.0e}_bs{r['batch_size']}_fs{r['features_start']}"
        print(f"{r.get('model',''):16s} {r['loss']:18s} {cfg:28s} {t:>8s}")

    write_model_loss_matrix(ranked)


# Fixed display order (ablation ladder) so the matrix reads simple -> complex.
MODEL_ORDER = ["basic_unet", "residual_unet", "attention_unet", "unetpp", "resunetpp"]
LOSS_ORDER = ["ce", "dice", "ce_dice", "weighted_ce_dice", "focal_dice",
              "cf_v", "cf_b", "cf_vb", "cbav", "cldice"]


def write_model_loss_matrix(best_rows):
    """Pivot the best-per-(model,loss) test AV Dice into a model x loss matrix."""
    cell = {(r.get("model", ""), r["loss"]): r["test_av_dice"] for r in best_rows}
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in cell)]
    losses = [l for l in LOSS_ORDER if any(k[1] == l for k in cell)]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else ""

    out_path = RESULTS_DIR / "model_loss_matrix.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model \\ loss"] + losses + ["model_mean"])
        for m in models:
            vals = [cell.get((m, l), "") for l in losses]
            nums = [v for v in vals if isinstance(v, (int, float))]
            mean = sum(nums) / len(nums) if nums else ""
            writer.writerow([m] + [fmt(v) for v in vals] + [fmt(mean)])
        # Per-loss mean across models (last row).
        loss_means = []
        for l in losses:
            nums = [cell[(m, l)] for m in models
                    if isinstance(cell.get((m, l)), (int, float))]
            loss_means.append(fmt(sum(nums) / len(nums)) if nums else "")
        writer.writerow(["loss_mean"] + loss_means + [""])

    print("\nModel x loss matrix (test AV Dice) -", out_path)
    header = f"{'model \\ loss':16s}" + "".join(f"{l[:9]:>10s}" for l in losses)
    print(header)
    for m in models:
        row = f"{m:16s}" + "".join(f"{fmt(cell.get((m, l), '')):>10s}" for l in losses)
        print(row)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", type=str, default="study", choices=["study", "coordinate", "grid"],
                   help="study = recorded 90-run design; coordinate = one axis at a time; "
                        "grid = full factorial over --axes.")
    p.add_argument("--axes", type=str, default=",".join(ALL_AXES),
                   help=f"Comma-separated axes to sweep. Options: {','.join(ALL_AXES)}")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--datasets", nargs="+", default=["drive", "les_av", "fundus_avseg"],
                   help="Datasets each run trains/selects/reports on (passed through "
                        "to the training script).")
    p.add_argument("--force", action="store_true",
                   help="Re-run configs even if test_metrics.json already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned runs and exit without training.")
    return p.parse_args()


def main():
    args = parse_args()
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown = [a for a in axes if a not in AXIS_VALUES]
    if unknown:
        raise SystemExit(f"Unknown axes: {unknown}. Valid: {ALL_AXES}")

    configs = build_configs(axes, args.mode)
    for cfg in configs:                      # tag every run with its dataset selection
        cfg["datasets"] = tuple(args.datasets)
    print(f"Planned {len(configs)} run(s) | mode={args.mode} | axes={axes} | "
          f"datasets={args.datasets} (epochs={args.epochs}, patience={args.patience}):")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i:2d}. {run_name_for(cfg)}")

    if args.dry_run:
        print("\n[dry-run] No training launched.")
        return

    failures = []
    for cfg in configs:
        ok = run_one(cfg, args.epochs, args.patience, args.force)
        if not ok:
            failures.append(run_name_for(cfg))

    print("\nAll runs attempted. Aggregating results...")
    write_sweep_summary(configs)

    if failures:
        print(f"\n{len(failures)} run(s) FAILED:")
        for name in failures:
            print("  -", name)
        sys.exit(1)


if __name__ == "__main__":
    main()
