param(
    [int]$Queries = 100,
    [int[]]$Seeds = @(20260607, 20260608),
    [int]$ViewCount = 6,
    [double]$ViewFraction = 0.6,
    [int]$SupportDepth = 20,
    [string]$Profile = "deepalimohapatra1973"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modal = Join-Path $repoRoot ".venv_modal\Scripts\modal.exe"
$logDir = Join-Path $repoRoot "runs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:MODAL_PROFILE = $Profile
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$modelSpecs = @(
    "fair_final_seed7101_best=/cache/models/arxiv-6_layer-model-jigsaw_coverage_final_ablation_final_seed7101_best_fullcov.pth",
    "fair_final_seed7102_best=/cache/models/arxiv-6_layer-model-jigsaw_coverage_final_ablation_final_seed7102_best_fullcov.pth"
) -join ";"
$fractionTag = ("{0:g}" -f $ViewFraction).Replace(".", "p")

$launched = foreach ($seed in $Seeds) {
    $outputTag = "retrieval_arxiv_khop_multiview_v${ViewCount}_f${fractionTag}_d${SupportDepth}_q${Queries}_seed${seed}"
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $logDir "${outputTag}_${timestamp}.out.log"
    $stderr = Join-Path $logDir "${outputTag}_${timestamp}.err.log"
    $process = Start-Process `
        -FilePath $modal `
        -ArgumentList @(
            "run", "--detach", "modal_retrieval_benchmark.py",
            "--model-specs", $modelSpecs,
            "--output-tag", $outputTag,
            "--queries", "$Queries",
            "--seed", "$seed",
            "--multi-view-only",
            "--multi-view-count", "$ViewCount",
            "--multi-view-fraction", "$ViewFraction",
            "--multi-view-support-depth", "$SupportDepth"
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
        Models = 2
        Views = $ViewCount
        ViewFraction = $ViewFraction
        SupportDepth = $SupportDepth
        ProcessId = $process.Id
        Stdout = $stdout
        Stderr = $stderr
    }
}

$launched | Format-Table -AutoSize
