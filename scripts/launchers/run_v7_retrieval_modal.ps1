param(
    [int]$Queries = 100,
    [int[]]$Seeds = @(20260607, 20260608)
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
$modelSpecs = @(
    "v6final=/cache/models/arxiv-6_layer-model-jigsaw_coverage_v6_cvar_livepos_from_v2.pth",
    "v7continue_best=/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth",
    "v7continue_final=/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6.pth",
    "v7clean_best=/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_clean_seed7002_best_fullcov.pth",
    "v7clean_final=/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_clean_seed7002.pth"
) -join ";"

$launched = foreach ($seed in $Seeds) {
    $outputTag = "retrieval_arxiv_khop_v7_candidates_q${Queries}_seed${seed}"
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
        ProcessId = $process.Id
        Stdout = $stdout
        Stderr = $stderr
    }
}

$launched | Format-Table -AutoSize
