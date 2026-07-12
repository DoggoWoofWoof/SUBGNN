param(
    [ValidateSet("arxiv", "cora")]
    [string]$Dataset = "arxiv",

    [int]$TrainingSeed = 7101,

    [string]$Profile = "darkphoenix696969696969",

    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modal = Join-Path $repoRoot ".venv_modal\Scripts\modal.exe"
$logDir = Join-Path $repoRoot "runs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:MODAL_PROFILE = $Profile
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$runName = "graphsage_contrastive_${Dataset}_seed${TrainingSeed}"
$arguments = @(
    "run", "--detach", "scripts/modal_train_graphsage.py",
    "--dataset", $Dataset,
    "--epochs", "90",
    "--steps-per-epoch", "100",
    "--batch-size", "8",
    "--run-name", $runName,
    "--learning-rate", "5e-5",
    "--min-learning-rate", "5e-6",
    "--warmup-steps", "100",
    "--scheduler-type", "cosine",
    "--gamma-partition", "0.0",
    "--gamma-fine-partition", "0.0",
    "--coverage-topk", "0",
    "--coverage-topk-weight", "0.0",
    "--max-live-positive-parts", "0",
    "--prob-k-hop", "0.55",
    "--prob-single-part", "0.10",
    "--prob-multi-coarse", "0.30",
    "--max-gpos-nodes", "2500",
    "--max-train-coarse-parts", "50",
    "--cache-refresh-steps", "20",
    "--validation-queries", "50",
    "--validation-interval", "5",
    "--validation-seeds", "31415,27182",
    "--validation-topks", "20,50,100",
    "--early-stopping-patience", "0",
    "--training-seed", $TrainingSeed
)

if (-not $Resume) {
    $arguments += "--fresh"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logDir "${runName}_${timestamp}.out.log"
$stderr = Join-Path $logDir "${runName}_${timestamp}.err.log"
$process = Start-Process `
    -FilePath $modal `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    Dataset = $Dataset
    TrainingSeed = $TrainingSeed
    Profile = $Profile
    OptimizerSteps = 9000
    Objective = "standard hierarchical InfoNCE (no coverage loss)"
    RunName = $runName
    ProcessId = $process.Id
    Stdout = $stdout
    Stderr = $stderr
} | Format-Table -AutoSize
