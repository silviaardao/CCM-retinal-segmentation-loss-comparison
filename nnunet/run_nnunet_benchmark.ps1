# Driver for nnU-Net v2 (isolated Python environment).
# Isensee et al. (2021), DOI: 10.1038/s41592-020-01008-z.
# Official code: https://github.com/MIC-DKFZ/nnUNet
#
#   .\nnunet\run_nnunet_benchmark.ps1 preprocess              # CPU; safe any time
#   .\nnunet\run_nnunet_benchmark.ps1 train                   # GPU; trains -Folds (default 0,1,2)
#   .\nnunet\run_nnunet_benchmark.ps1 train   -Folds 3,4      # GPU; just the missing folds
#   .\nnunet\run_nnunet_benchmark.ps1 predict -Folds 0,1,2    # GPU; ensemble of the given folds
#
# On the Windows system used for these runs, the data-augmentation pool raised:
#   "RuntimeError: One or more background workers are no longer alive"
# This launcher uses in-process augmentation and single-process prediction.
#
# Prediction defaults to folds 0,1,2, matching the completed local configuration.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('preprocess', 'train', 'predict', 'all')]
    [string]$Step,

    # Folds to train and ensemble. Both steps default to the three folds used by
    # the completed benchmark so the `all` mode remains internally consistent.
    [int[]]$Folds,

    [string]$NnUNetScripts = $env:NNUNET_SCRIPTS
)
$ErrorActionPreference = 'Stop'

$PROJECT = Split-Path -Parent $PSScriptRoot
$env:nnUNet_raw = "$PROJECT\nnunet\nnUNet_raw"
$env:nnUNet_preprocessed = "$PROJECT\nnunet\nnUNet_preprocessed"
$env:nnUNet_results = "$PROJECT\nnunet\nnUNet_results"

# Run augmentation in the main process to avoid the Windows worker failure.
$env:nnUNet_n_proc_DA = '0'

$S = if ($NnUNetScripts) { $NnUNetScripts } else {
    Split-Path -Parent (Get-Command nnUNetv2_train.exe -ErrorAction Stop).Source
}
$DATASET_ID = 1
$CONFIG = '2d'                # 2D fundus images

if ($Step -in 'preprocess', 'all') {
    Write-Host '=== plan & preprocess ===' -ForegroundColor Cyan
    & "$S\nnUNetv2_plan_and_preprocess.exe" -d $DATASET_ID --verify_dataset_integrity
    if ($LASTEXITCODE -ne 0) { throw "preprocess failed (exit $LASTEXITCODE)" }
    Copy-Item "$PROJECT\nnunet\splits_final.json" `
        "$env:nnUNet_preprocessed\Dataset001_RetinaAV\splits_final.json" -Force
    Write-Host 'Installed the committed cross-validation split.' -ForegroundColor Green
}

if ($Step -in 'train', 'all') {
    $trainFolds = if ($Folds) { $Folds } else { 0, 1, 2 }
    $failed = @()
    foreach ($f in $trainFolds) {
        Write-Host "=== training fold $f (1000 epochs) ===" -ForegroundColor Cyan
        # Don't let one bad fold abort the rest; collect and report at the end.
        $ErrorActionPreference = 'Continue'
        & "$S\nnUNetv2_train.exe" $DATASET_ID $CONFIG $f
        $code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($code -ne 0) {
            Write-Host "  FOLD $f FAILED (exit $code)" -ForegroundColor Red
            $failed += $f
        }
    }
    if ($failed.Count -gt 0) {
        throw "Folds failed: $($failed -join ', '). Check the DA-worker error above."
    }
    Write-Host "All folds trained: $($trainFolds -join ', ')" -ForegroundColor Green
}

if ($Step -in 'predict', 'all') {
    $predFolds = if ($Folds) { $Folds } else { 0, 1, 2 }
    Write-Host "=== predict on held-out test set (folds: $($predFolds -join ',')) ===" -ForegroundColor Cyan
    & "$S\nnUNetv2_predict.exe" `
        -i "$env:nnUNet_raw\Dataset001_RetinaAV\imagesTs" `
        -o "$PROJECT\nnunet\predictions_test" `
        -d $DATASET_ID -c $CONFIG -f $predFolds `
        -npp 1 -nps 1                     # single-process pre/post - Windows fix
    if ($LASTEXITCODE -ne 0) { throw "predict failed (exit $LASTEXITCODE)" }
    Write-Host "Now score it: python scripts\evaluate_nnunet.py" -ForegroundColor Green
}
