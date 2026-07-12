param(
    [int]$Queries = 100,
    [int[]]$Seeds = @(20260607, 20260608),
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

$variants = @(
    "control_seed7101",
    "control_seed7102",
    "cvar_seed7101",
    "topk_seed7101",
    "final_seed7101",
    "final_seed7102"
)
$modelSpecs = foreach ($variant in $variants) {
    "fair_${variant}_best=/cache/models/arxiv-6_layer-model-jigsaw_coverage_final_ablation_${variant}_best_fullcov.pth"
    "fair_${variant}_final=/cache/models/arxiv-6_layer-model-jigsaw_coverage_final_ablation_${variant}.pth"
}
$modelSpecs = $modelSpecs -join ";"

$launched = foreach ($seed in $Seeds) {
    $outputTag = "retrieval_arxiv_khop_fair_ablation_q${Queries}_seed${seed}"
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
        Profile = $Profile
        Models = $variants.Count * 2
        ProcessId = $process.Id
        Stdout = $stdout
        Stderr = $stderr
    }
}

$launched | Format-Table -AutoSize
