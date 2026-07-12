param(
    [ValidateSet("continuation", "clean")]
    [string]$Mode = "continuation"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modal = Join-Path $repoRoot ".venv_modal\Scripts\modal.exe"
$logDir = Join-Path $repoRoot "runs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if ((& $modal profile current).Trim() -ne "pilgnnteam") {
    & $modal profile activate pilgnnteam
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$common = @(
    "run", "--detach", "scripts/train_jigsaw_model.py",
    "--dataset", "arxiv",
    "--batch-size", "8",
    "--steps-per-epoch", "100",
    "--gamma-partition", "1.0",
    "--gamma-fine-partition", "0.0",
    "--coverage-temperature", "0.05",
    "--coverage-topk", "20",
    "--coverage-topk-weight", "0.35",
    "--coverage-topk-margin", "0.0",
    "--coverage-positive-aggregation", "cvar",
    "--coverage-cvar-fraction", "0.25",
    "--max-live-positive-parts", "24",
    "--prob-k-hop", "0.55",
    "--prob-single-part", "0.10",
    "--prob-multi-coarse", "0.30",
    "--max-gpos-nodes", "2500",
    "--max-train-coarse-parts", "50",
    "--cache-refresh-steps", "20",
    "--scheduler-type", "cosine",
    "--validation-queries", "50",
    "--validation-interval", "2",
    "--validation-seeds", "31415,27182",
    "--early-stopping-patience", "4"
)

if ($Mode -eq "continuation") {
    $runName = "coverage_v7_continue_from_v6"
    $arguments = $common + @(
        "--epochs", "16",
        "--run-name", $runName,
        "--resume-from-checkpoint", "/cache/models/arxiv-6_layer-model-jigsaw_coverage_v6_cvar_livepos_from_v2.pth",
        "--resume-model-only",
        "--learning-rate", "1e-5",
        "--min-learning-rate", "2e-6",
        "--warmup-steps", "0",
        "--training-seed", "7001"
    )
} else {
    $runName = "coverage_v7_clean_seed7002"
    $arguments = $common + @(
        "--epochs", "40",
        "--run-name", $runName,
        "--fresh",
        "--learning-rate", "5e-5",
        "--min-learning-rate", "5e-6",
        "--warmup-steps", "100",
        "--training-seed", "7002"
    )
}

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
    Mode = $Mode
    RunName = $runName
    ProcessId = $process.Id
    Stdout = $stdout
    Stderr = $stderr
} | Format-Table -AutoSize
