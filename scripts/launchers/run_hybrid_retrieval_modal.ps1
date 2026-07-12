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
$stdout = Join-Path $logDir "retrieval_hybrid_consensus_v2_q30_seed42_${timestamp}.out.log"
$stderr = Join-Path $logDir "retrieval_hybrid_consensus_v2_q30_seed42_${timestamp}.err.log"
$modelSpecs = "v2=/cache/models/arxiv-6_layer-model-jigsaw_coverage_v2_allpos_fresh.pth"

$process = Start-Process `
    -FilePath $modal `
    -ArgumentList @(
        "run", "--detach", "modal_retrieval_benchmark.py",
        "--model-specs", $modelSpecs,
        "--output-tag", "retrieval_arxiv_khop_hybrid_consensus_v2_q30_seed42",
        "--queries", "30",
        "--seed", "42"
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
