# Compare seven objectives inside one matched nnU-Net setting.
#
# Every job uses Dataset001, the 2d configuration, fold 0 and 100 epochs. The
# custom trainers change loss construction only; nnU-Net continues to provide
# its network, preprocessing, augmentation, schedule and deep supervision.
#
# nnU-Net: Isensee et al. (2021), DOI: 10.1038/s41592-020-01008-z.
# CF-Loss: Zhou et al. (2024), DOI: 10.1016/j.media.2024.103098.
# clDice: Shit et al. (2021), DOI: 10.1109/CVPR46437.2021.01629.

param(
    [string]$Python = $env:RETINA_AV_PYTHON,
    [string]$NnUNetScripts = $env:NNUNET_SCRIPTS
)

$ErrorActionPreference = "Continue"
$PROJECT = Split-Path -Parent $PSScriptRoot
$METRIC_PYTHON = if ($Python) { $Python } else { (Get-Command python -ErrorAction Stop).Source }
$VENV = if ($NnUNetScripts) { $NnUNetScripts } else {
    Split-Path -Parent (Get-Command nnUNetv2_train.exe -ErrorAction Stop).Source
}
$NNUNET_PYTHON = "$VENV\python.exe"
Set-Location $PROJECT
$env:RETINA_AV_PROJECT_ROOT = $PROJECT
$env:nnUNet_raw = "$PROJECT\nnunet\nnUNet_raw"
$env:nnUNet_preprocessed = "$PROJECT\nnunet\nnUNet_preprocessed"
$env:nnUNet_results = "$PROJECT\nnunet\nnUNet_results"
$env:nnUNet_n_proc_DA = "0"

$DS = 1
$CFG = "2d"
$FOLD = 0
$RESULTS = "$env:nnUNet_results\Dataset001_RetinaAV"
$IMAGESTS = "$env:nnUNet_raw\Dataset001_RetinaAV\imagesTs"
$PREDROOT = "$PROJECT\nnunet\predictions_cf"
$LogDir = "$PROJECT\results\queue_logs"
New-Item -ItemType Directory -Force -Path $LogDir, $PREDROOT | Out-Null
$Master = "$LogDir\nnunet_objectives_$(Get-Date -Format yyyyMMdd_HHmmss).log"
$Lock = "$LogDir\nnunet_objectives.lock"

function Say($Message) {
    $Line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    Write-Host $Line
    Add-Content $Master $Line
}

function Test-CompletePredictionSet($PredictionDirectory) {
    if (-not (Test-Path $PredictionDirectory)) { return $false }
    $Expected = @(Get-ChildItem $IMAGESTS -Filter "*_0000.png" -ErrorAction SilentlyContinue |
        ForEach-Object { $_.BaseName -replace "_0000$", "" } | Sort-Object)
    $Actual = @(Get-ChildItem $PredictionDirectory -Filter "*.png" -ErrorAction SilentlyContinue |
        ForEach-Object { $_.BaseName } | Sort-Object)
    if ($Expected.Count -ne 44) {
        throw "Expected 44 test inputs in $IMAGESTS, found $($Expected.Count)."
    }
    if ($Actual.Count -ne $Expected.Count) { return $false }
    return (@(Compare-Object $Expected $Actual).Count -eq 0)
}

if (Test-Path $Lock) {
    $ExistingPid = Get-Content $Lock -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ExistingPid -and (Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue)) {
        Write-Host "The nnU-Net objective comparison is already running (pid $ExistingPid)."
        exit 2
    }
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}
$PID | Out-File $Lock -Encoding ascii -Force

$Jobs = @(
    @{ label = "default"; tr = "nnUNetTrainer_100epochs" },
    @{ label = "ce";      tr = "nnUNetTrainerCELoss_100epochs" },
    @{ label = "dice";    tr = "nnUNetTrainerDiceLoss_100epochs" },
    @{ label = "cf_v";    tr = "nnUNetTrainerCFv_100epochs" },
    @{ label = "cf_b";    tr = "nnUNetTrainerCFb_100epochs" },
    @{ label = "cf_vb";   tr = "nnUNetTrainerCFvb_100epochs" },
    @{ label = "cldice";  tr = "nnUNetTrainerCLDice_100epochs" }
)

$Failures = @()
try {
    $CommittedSplit = "$PROJECT\nnunet\splits_final.json"
    $ActiveSplit = "$env:nnUNet_preprocessed\Dataset001_RetinaAV\splits_final.json"
    if (-not (Test-Path $CommittedSplit)) { throw "Missing committed split: $CommittedSplit" }
    if (-not (Test-Path (Split-Path -Parent $ActiveSplit))) {
        throw "nnU-Net preprocessing is missing. Run run_nnunet_benchmark.ps1 preprocess first."
    }
    Copy-Item $CommittedSplit $ActiveSplit -Force

    Say "Installing the project custom trainers into the active nnU-Net environment."
    & $NNUNET_PYTHON "$PROJECT\scripts\install_nnunet_custom_trainers.py" *>&1 |
        Tee-Object "$LogDir\nnunet_objectives_install.log" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Custom trainer installation failed (exit $LASTEXITCODE)." }

    Say "Starting $($Jobs.Count) objectives (fold $FOLD, 100 epochs)."
    foreach ($Job in $Jobs) {
        $Trainer = $Job.tr
        $Label = $Job.label
        $Checkpoint = "$RESULTS\${Trainer}__nnUNetPlans__${CFG}\fold_$FOLD\checkpoint_final.pth"
        $PredictionDirectory = "$PREDROOT\$Label"

        if (Test-Path $Checkpoint) {
            Say "SKIP train $Label (final checkpoint exists)."
        } else {
            Say "TRAIN $Label ($Trainer)."
            & "$VENV\nnUNetv2_train.exe" $DS $CFG $FOLD -tr $Trainer *>&1 |
                Tee-Object "$LogDir\nnunet_objectives_train_$Label.log"
            if ($LASTEXITCODE -ne 0) {
                Say "FAILED train $Label (exit $LASTEXITCODE)."
                $Failures += "train:$Label"
                continue
            }
        }

        if (Test-CompletePredictionSet $PredictionDirectory) {
            Say "SKIP predict $Label (all 44 predictions exist)."
        } else {
            Say "PREDICT $Label. Existing partial output, if any, will be completed or replaced by nnU-Net."
            & "$VENV\nnUNetv2_predict.exe" -i $IMAGESTS -o $PredictionDirectory -d $DS -c $CFG `
                -f $FOLD -tr $Trainer -npp 1 -nps 1 *>&1 |
                Tee-Object "$LogDir\nnunet_objectives_predict_$Label.log"
            if ($LASTEXITCODE -ne 0) {
                Say "FAILED predict $Label (exit $LASTEXITCODE)."
                $Failures += "predict:$Label"
                continue
            }
            if (-not (Test-CompletePredictionSet $PredictionDirectory)) {
                Say "FAILED predict $Label (output is not a complete 44-case set)."
                $Failures += "incomplete:$Label"
            }
        }
    }

    if ($Failures.Count -gt 0) {
        throw "Failed steps: $($Failures -join ', ')"
    }

    Say "Scoring the seven complete prediction sets."
    & $METRIC_PYTHON "$PROJECT\scripts\score_cf_experiment.py" *>&1 |
        Tee-Object "$LogDir\nnunet_objectives_score.log"
    if ($LASTEXITCODE -ne 0) { throw "Scoring failed (exit $LASTEXITCODE)." }
    Say "Done. See results\nnunet_cf_experiment\summary.csv."
} finally {
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}
