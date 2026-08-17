# How nnU-Net is used in this project

The U-Net models in `models/` are created directly by the main Python training script.
nnU-Net is instead installed as a separate framework because it manages its own data format,
preprocessing, network configuration, training, and prediction.

This project uses nnU-Net in four steps: convert the retinal data to its required format,
copy the custom loss trainers into the installed framework, run the nnU-Net commands, and
score the predictions with the study's metrics. The PowerShell files simply run those
commands in the correct order.

## What is in this folder

```text
custom_trainers/
  nnUNetTrainerCF.py       CF-V, CF-B, CF-VB, CE and Dice trainer variants
  nnUNetTrainerCLDice.py   soft-clDice trainer variant

splits_final.json           exact five-fold cross-validation assignment
run_nnunet_benchmark.ps1   longer three-fold nnU-Net benchmark
run_nnunet_objectives.ps1  one-fold, 100-epoch seven-objective comparison
```

The custom trainers change the loss construction only. They retain nnU-Net's network,
preprocessing, augmentation, optimizer, learning-rate schedule, and deep-supervision
handling. The CF configurations use the source coefficients: vessel-density beta=1.0 and
box-count lambda=0.5.

These are nnU-Net-specific adaptations, not exact copies of the controlled U-Net objectives.
The CF trainers use unweighted cross-entropy because static project class weights are not
passed into the trainer. The soft-clDice trainer uses nnU-Net's unweighted Dice and
cross-entropy base plus a 0.1 soft-clDice term. The controlled U-Net equivalents use the
study class weights where documented. Comparing the rankings is still useful, but it tests
how related structural terms behave inside a different optimisation pipeline.

## Preparing the environment and data

Create and activate a second environment, then install the recorded nnU-Net version:

```powershell
python -m venv .venv-nnunet
.venv-nnunet\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements\nnunet.txt
```

From the repository root, convert the organized AV data into nnU-Net's format:

```powershell
python scripts\convert_to_nnunet.py
```

This creates `nnunet/nnUNet_raw/Dataset001_RetinaAV/`. The main study's training and
validation samples form nnU-Net's training pool; the same held-out test samples remain the
test set. nnU-Net requires a separate cross-validation assignment inside that training pool.
The exact assignment used by the project is committed as `nnunet/splits_final.json`; the
benchmark and objective launchers copy it into the preprocessed dataset before training.

## Custom trainer installation

nnU-Net discovers trainer classes from inside its installed package. The project copy remains
the editable source, and this command copies it into the active nnU-Net environment:

```powershell
$env:RETINA_AV_PROJECT_ROOT = (Get-Location).Path
python scripts\install_nnunet_custom_trainers.py
```

The installer also checks that nnU-Net can find every trainer class. The launchers run this
installation step again before the custom experiments so the installed copy is current.

## Longer nnU-Net benchmark

The general launcher accepts three steps:

```powershell
.\nnunet\run_nnunet_benchmark.ps1 preprocess
.\nnunet\run_nnunet_benchmark.ps1 train
.\nnunet\run_nnunet_benchmark.ps1 predict -Folds 0,1,2
python scripts\evaluate_nnunet.py
```

## Reduced objective comparison

`run_nnunet_objectives.ps1` runs fold 0 for 100 epochs with the default nnU-Net objective,
CE, Dice, CF-V, CF-B, CF-VB and soft-clDice. Predictions are stored under
`nnunet/predictions_cf/` and scored by
`scripts/score_cf_experiment.py` using the same metric implementation as the main study.

The reduced experiment predicts from each trainer's final 100-epoch checkpoint. This differs
from the controlled U-Net study, which retains the checkpoint with the highest validation AV
Dice. The distinction is part of the experimental design and should be retained in any report.

The objective launcher is restartable: an existing final checkpoint skips training, and
prediction is skipped only when the exact 44 expected test files are present. The scoring
script also rejects missing or unexpected cases and stores each image identifier in the
per-image CSV. A lock file prevents two copies from running simultaneously and is removed in
a `finally` block after success or failure.

## References

- Isensee et al. (2021), nnU-Net, https://doi.org/10.1038/s41592-020-01008-z.
- Zhou et al. (2024), CF-Loss, https://doi.org/10.1016/j.media.2024.103098.
- Shit et al. (2021), clDice, https://doi.org/10.1109/CVPR46437.2021.01629.
