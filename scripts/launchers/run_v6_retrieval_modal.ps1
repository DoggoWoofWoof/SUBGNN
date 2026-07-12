param(
    [int]$Queries = 30,
    [int]$Seed = 42,
    [string]$OutputTag = "retrieval_arxiv_khop_hybrid_consensus_v2_v6_q30_seed42",
    [switch]$IncludeBest
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
$stdout = Join-Path $logDir "${OutputTag}_${timestamp}.out.log"
$stderr = Join-Path $logDir "${OutputTag}_${timestamp}.err.log"
$models = @(
    "v2=/cache/models/arxiv-6_layer-model-jigsaw_coverage_v2_allpos_fresh.pth"
    "v6final=/cache/models/arxiv-6_layer-model-jigsaw_coverage_v6_cvar_livepos_from_v2.pth"
)
if ($IncludeBest) {
    $models += "v6best=/cache/models/arxiv-6_layer-model-jigsaw_coverage_v6_cvar_livepos_from_v2_best_fullcov.pth"
}
$modelSpecs = $models -join ";"

$process = Start-Process `
    -FilePath $modal `
    -ArgumentList @(
        "run", "--detach", "modal_retrieval_benchmark.py",
        "--model-specs", $modelSpecs,
        "--output-tag", $OutputTag,
        "--queries", "$Queries",
        "--seed", "$Seed"
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
