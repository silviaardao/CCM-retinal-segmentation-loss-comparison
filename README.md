# Retinal artery-vein segmentation with U-Net architectures and loss functions

This repository contains the code from a loss function and U-Net comparison study
applied to retinal image segmentation. The aim was to understand how the choice of U-Net
architecture and loss function affects multiclass retinal artery-vein segmentation.

The task has five classes: background, artery, vein, artery-vein overlap, and ambiguous
vessel. The same training and evaluation pipeline was used for five U-Net-family architectures
and ten loss configurations. A separate comparison with nnU-Net was also run.

Compact final result tables are included under `results/summary/` so the code can be connected
to the principal report findings. Large generated material—checkpoints, logs, predictions,
training histories and figures—is not included.

## Original contribution

The original contribution of this project is the controlled comparison of U-Net-family
architectures and loss functions within one common five-class retinal artery-vein pipeline.
It also includes study-specific adaptations of structure-aware objectives and a comparison
with nnU-Net. The underlying architecture and loss concepts come from the publications and
software projects cited in the relevant source files and documentation.

## Project structure

```text
datasets/
  splits/              fixed train, validation, and test CSV files
  av_dataset.py        label conversion and paired image/mask augmentation
  manifest_dataset.py  loads the samples listed in the split CSV files
  preprocessing.py     organizes the three source datasets in one format

models/                 five U-Net-family architecture definitions

evaluation/
  README.md             explanation of the losses, metrics, and adaptations
  losses/               baseline and structure-aware loss functions
  metrics/              pixel and vessel-structure evaluation metrics

scripts/
  preprocess_datasets.py   prepares the common dataset and split files
  train_av_multiclass.py   trains one model/loss configuration
  sweep_multiclass.py      launches the planned architecture, loss, and seed runs
  convert_to_nnunet.py     converts the same data for nnU-Net
  evaluate_nnunet.py       evaluates the longer nnU-Net benchmark
  score_cf_experiment.py   evaluates the reduced nnU-Net objective comparison
  install_nnunet_custom_trainers.py  makes the custom trainers visible to nnU-Net

nnunet/
  custom_trainers/      CF-Loss and soft-clDice trainer extensions
  splits_final.json     exact five-fold split used by nnU-Net
  run_nnunet_benchmark.ps1   longer three-fold benchmark
  run_nnunet_objectives.ps1  reduced seven-objective comparison

requirements/           dependencies for the main study and nnU-Net
results/summary/         compact tables for the final reported experiments
```

Raw images, masks, model checkpoints, predictions, logs, and figures are deliberately
excluded because they are large or cannot be redistributed. The CSV files in
`datasets/splits/` are kept so the exact sample assignment is recorded.

## Principal results

The complete controlled matrix, five-seed Basic U-Net summary, reduced nnU-Net objective
comparison and full nnU-Net benchmark are available in [`results/summary/`](results/README.md).
These tables are intentionally compact; per-image outputs and training artefacts remain
outside this release.

The controlled U-Net and nnU-Net objectives share the same five-class task, but they are not
identical complete formulas. The custom nnU-Net CF trainers use unweighted cross-entropy, and
the nnU-Net soft-clDice trainer uses unweighted Dice plus cross-entropy as its base. The local
controlled versions use the study class weights where stated in `evaluation/README.md`.

## Study design

The architecture comparison contains:

- basic U-Net;
- residual U-Net;
- attention U-Net;
- U-Net++;
- ResUNet++.

The loss comparison contains cross-entropy, Dice, CE and Dice combinations, focal Dice,
three CF-Loss variants, the study's CBAV-Loss adaptation, and soft-clDice. Citations and
implementation notes are included in the corresponding model and loss source files. A
plain-language overview is provided in [`evaluation/README.md`](evaluation/README.md).

The training split updates the model weights. The validation split selects the checkpoint,
controls learning-rate reduction, and determines early stopping. The test split is held out
until the selected checkpoint is evaluated at the end of training.

### Controlled U-Net experiment configuration

The settings below apply to the 90-run controlled U-Net comparison and were verified
against the saved configuration files from those completed runs. They do not describe the
nnU-Net experiments, which used nnU-Net's own preprocessing, architecture, optimizer and
learning-rate schedule, with separate training budgets documented in `nnunet/README.md`.

| Setting | Value |
| --- | --- |
| Input size | 512 x 512 pixels |
| Initial feature width | 32 |
| Learning rate | 0.0004 |
| Batch size | 2 |
| Maximum epochs | 300 |
| Early-stopping patience | 50 epochs |
| Optimizer | Adam, beta1=0.5 and beta2=0.999 |
| Multi-seed values | 42, 123, 456, 7 and 2024 |

Within this controlled pipeline, the five architectures and ten losses were compared at
seed 42. The four additional seeds were used only for the Basic U-Net loss comparison.
Dataset membership is fixed by the three split manifests in `datasets/splits/`.

The multi-seed analysis therefore estimates loss-function variability for Basic U-Net. The
architecture comparison uses one fixed seed and should not be interpreted as an estimate of
architecture-level variance.

## Installation

The main experiment and nnU-Net use separate Python environments. This avoids changing the
main experiment when nnU-Net installs its own dependencies.

For the main experiment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install a PyTorch build suitable for the computer from the
[official PyTorch installation page](https://pytorch.org/get-started/locally/) and install
the remaining packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements/main.txt
```

The code automatically uses CUDA when available, Apple MPS when available, and otherwise
the CPU. Full training is much faster on a GPU.

The main requirement file records compatible version ranges because the exact package export
from the original training computer was not saved. The nnU-Net comparison records the exact
framework release (`nnunetv2==2.8.1`).

## Data preparation

The datasets are not included. After downloading them from the sources listed below, the
preprocessing code expects this local layout:

```text
data/
  training/images/ and training/av/       DRIVE/RITE training data
  test/images/ and test/av/               DRIVE/RITE test data
  raw/les_av/images/
  raw/les_av/arteries-and-veins/
  raw/fundus_avseg/images/
  raw/fundus_avseg/annotation/
  raw/fundus_avseg/training.txt
  raw/fundus_avseg/testing.txt
```

Prepare all three datasets with:

```bash
python scripts/preprocess_datasets.py --dataset all
```

The original image and colour mask are copied without changing their pixels. The colour mask
is converted to class indices in memory when it is loaded for training. Preprocessing writes
the organized local data under `data/organized/` and records the sample paths in
`datasets/splits/`.

## Running the main experiment

A single short run can be used to check that the installation and data paths work:

```bash
python scripts/train_av_multiclass.py --model basic_unet --loss ce_dice --epochs 2 --patience 2 --run-name installation_check
```

A normal single configuration uses the same script with the required architecture, loss,
seed, and training budget. For example:

```bash
python scripts/train_av_multiclass.py --model basic_unet --loss cf_vb --epochs 300 --seed 42 --run-name basic_unet_cf_vb_seed42
```

Outputs are written to `results/multiclass/<run-name>/`. Each run records its configuration,
training history, validation-selected checkpoint, test metrics, per-image metrics, and two
simple visual checks.

The sweep launcher's default `study` mode reproduces the recorded 90-run design and skips
completed runs. Always inspect the planned configurations before starting a large experiment:

```bash
python scripts/sweep_multiclass.py --dry-run
```

A compact check of the models, losses, label conversion and split manifests can be run
without downloading the datasets:

```bash
python -m unittest tests.test_core
```

## nnU-Net experiment

nnU-Net is a separate segmentation framework that configures its own preprocessing,
architecture, training schedule, and cross-validation. The code in `nnunet/` does not
reimplement nnU-Net. It prepares this study's dataset, adds the custom loss trainers, and
calls the official nnU-Net commands. See [`nnunet/README.md`](nnunet/README.md) for the
plain-language explanation and commands.

## Dataset sources

The datasets are not included in this repository. They must be downloaded from their
original sources and used according to the terms given by each provider:

- **DRIVE fundus images:** [official DRIVE database](https://drive.grand-challenge.org/)
- **RITE artery-vein annotations for DRIVE:** [official RITE dataset page](https://eye.medicine.uiowa.edu/rite-dataset)
- **LES-AV:** [official Figshare dataset record](https://doi.org/10.6084/m9.figshare.11857698)
- **Fundus-AVSeg:** [official Figshare dataset record](https://doi.org/10.6084/m9.figshare.27938034)

## Software attribution

**nnU-Net:** Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., and Maier-Hein,
K. H. (2021). *nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image
Segmentation*. Nature Methods, 18, 203–211.
https://doi.org/10.1038/s41592-020-01008-z. Official implementation:
[MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet).

This repository contains study-specific implementations and adaptations. It does not claim
that these are the original authors' reference implementations.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Author

Silvia Rodríguez Ardao
