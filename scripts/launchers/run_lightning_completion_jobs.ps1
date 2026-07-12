param(
    [string]$Owner = "whenthedarknightrises",
    [string]$Teamspace = "financial-llm-training-project",
    [string]$PackageModel = "jigsaw-production-benchmark-package-v1",
    [string]$PackageDir = "runs\lightning_production_benchmark_package_v1",
    [string]$Machine = "CPU_X_8",
    [string]$Cloud = "gcp-lightning-public-prod",
    [switch]$FetchArxivV7,
    [switch]$SkipUpload,
    [switch]$SkipCora,
    [switch]$SkipArxivNonModel,
    [switch]$SkipArxivV7NegFill
)

$ErrorActionPreference = "Stop"

if (-not $env:LIGHTNING_USER_ID -or -not $env:LIGHTNING_API_KEY) {
    throw "Set LIGHTNING_USER_ID and LIGHTNING_API_KEY before launching Lightning jobs."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv_modal\Scripts\python.exe"
$launcher = Join-Path $repoRoot "scripts\lightning_production_benchmark.py"
$v7Best = Join-Path $repoRoot "models\arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth"

function Invoke-Launcher {
    param([string[]]$LauncherArgs)
    & $python $launcher @LauncherArgs
    if ($LASTEXITCODE -ne 0) {
        throw "lightning_production_benchmark.py failed: $($LauncherArgs -join ' ')"
    }
}

if ($FetchArxivV7) {
    & (Join-Path $repoRoot "scripts\fetch_modal_arxiv_v7_models.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Arxiv v7 fetch failed."
    }
}

Invoke-Launcher @(
    "prepare-package",
    "--package-dir", $PackageDir
)

if (-not $SkipUpload) {
    Invoke-Launcher @(
        "upload-package",
        "--package-dir", $PackageDir,
        "--owner", $Owner,
        "--teamspace", $Teamspace,
        "--package-model", $PackageModel
    )
}

$common = @(
    "--owner", $Owner,
    "--teamspace", $Teamspace,
    "--package-model", $PackageModel,
    "--machine", $Machine,
    "--cloud", $Cloud,
    "--interruptible",
    "--workers", "1",
    "--parallel-mode", "task",
    "--queries", "50",
    "--target-sizes", "20,50,100",
    "--seeds", "20260607,20260608",
    "--signature", "type_feat32",
    "--solver-timeout", "5"
)

if (-not $SkipCora) {
    $launchArgs = @(
        "launch-job"
    ) + $common + @(
        "--job-name", "jigsaw-cora-clean-full-gcp-cpux8-v1",
        "--result-model", "jigsaw-cora-clean-full-gcp-cpux8-v1-results",
        "--dataset", "cora",
        "--methods", "neural_component,random_component,mean_feature_component,mean_rrf_component,topo_feature_component,filterall_component",
        "--budgets", "2,5,10,20",
        "--full-budget", "20",
        "--model-specs", "cora_jigsaw=/workspace/jigsaw_pkg/cache/models/cora-6_layer-model-jigsaw.pth",
        "--output-prefix", "prod_cora_clean"
    )
    Invoke-Launcher $launchArgs
}

if (-not $SkipArxivNonModel) {
    $launchArgs = @(
        "launch-job"
    ) + $common + @(
        "--job-name", "jigsaw-arxiv-random-topo-filterall-gcp-cpux8-v1",
        "--result-model", "jigsaw-arxiv-random-topo-filterall-gcp-cpux8-v1-results",
        "--dataset", "arxiv",
        "--methods", "random_component,topo_feature_component,filterall_component",
        "--budgets", "20,50,100,200",
        "--full-budget", "200",
        "--output-prefix", "prod_arxiv_fill_rtf"
    )
    Invoke-Launcher $launchArgs
}

if (-not $SkipArxivV7NegFill) {
    if (-not (Test-Path -LiteralPath $v7Best)) {
        throw "Exact Arxiv v7 checkpoint is missing: $v7Best. Run with -FetchArxivV7 after Modal connectivity is available."
    }
    $launchArgs = @(
        "launch-job"
    ) + $common + @(
        "--job-name", "jigsaw-arxiv-v7-negfill-mean-meanrrf-gcp-cpux8-v1",
        "--result-model", "jigsaw-arxiv-v7-negfill-mean-meanrrf-gcp-cpux8-v1-results",
        "--dataset", "arxiv",
        "--methods", "mean_feature_component,mean_rrf_component",
        "--query-types", "negative",
        "--seeds", "20260608",
        "--budgets", "20,50,100,200",
        "--full-budget", "200",
        "--model-specs", "arxiv_v7_continue_best=/workspace/jigsaw_pkg/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth",
        "--output-prefix", "prod_arxiv_v7_negfill"
    )
    Invoke-Launcher $launchArgs
}

Write-Host "[DONE] Completion jobs submitted."
