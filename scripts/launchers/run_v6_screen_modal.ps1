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
$stdout = Join-Path $logDir "coverage_v6_cvar_livepos_from_v2_${timestamp}.out.log"
$stderr = Join-Path $logDir "coverage_v6_cvar_livepos_from_v2_${timestamp}.err.log"

$process = Start-Process `
    -FilePath $modal `
    -ArgumentList @(
        "run", "--detach", "scripts/train_jigsaw_model.py",
        "--dataset", "arxiv",
        "--epochs", "20",
        "--steps-per-epoch", "75",
        "--batch-size", "8",
        "--run-name", "coverage_v6_cvar_livepos_from_v2",
        "--resume-from-checkpoint", "/cache/models/arxiv-6_layer-model-jigsaw_coverage_v2_allpos_fresh.pth",
        "--resume-model-only",
        "--gamma-partition", "1.0",
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
        "--learning-rate", "5e-5",
        "--scheduler-type", "cosine",
        "--min-learning-rate", "1e-5",
        "--warmup-steps", "100",
        "--validation-queries", "30",
        "--validation-interval", "2",
        "--validation-seed", "31415"
    ) `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    ProcessId = $process.Id
    Stdout = $stdout
    Stderr = $stderr
} | Format-Table -AutoSize
