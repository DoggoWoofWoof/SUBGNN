param(
    [int]$Queries = 100,
    [int[]]$Seeds = @(20260607, 20260608),
    [string]$Profile = "darkphoenix696969696969",
    [string]$OutputPrefix = "retrieval_arxiv_khop_graphsage",
    [string]$ModelSpecs = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modal = Join-Path $repoRoot ".venv_modal\Scripts\modal.exe"
$logDir = Join-Path $repoRoot "runs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:MODAL_PROFILE = $Profile
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if (-not $ModelSpecs) {
    $ModelSpecs = @(
        "graphsage_best=/cache/models/arxiv-6_layer-model-graphsage_graphsage_contrastive_arxiv_seed7101_best_fullcov.pth",
        "graphsage_final=/cache/models/arxiv-6_layer-model-graphsage_graphsage_contrastive_arxiv_seed7101.pth"
    ) -join ";"
}

$launched = foreach ($seed in $Seeds) {
    $outputTag = "${OutputPrefix}_q${Queries}_seed${seed}"
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $logDir "${outputTag}_${timestamp}.out.log"
    $stderr = Join-Path $logDir "${outputTag}_${timestamp}.err.log"
    $process = Start-Process `
        -FilePath $modal `
        -ArgumentList @(
            "run", "--detach", "modal_retrieval_benchmark.py",
            "--model-specs", $ModelSpecs,
            "--output-tag", $outputTag,
            "--queries", "$Queries",
            "--target-sizes", "20",
            "--seed", "$seed"
        ) `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru

    [pscustomobject]@{
        Seed = $seed
        Queries = $Queries
        Profile = $Profile
        ProcessId = $process.Id
        Stdout = $stdout
        Stderr = $stderr
    }
}

$launched | Format-Table -AutoSize
