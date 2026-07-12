param(
    [string]$Owner = "whenthedarknightrises",
    [ValidateSet("user", "org")]
    [string]$OwnerKind = "user",
    [string]$Teamspace = "financial-llm-training-project",
    [string]$Version = "v2",
    [string]$PackageModel = "",
    [string]$PackageDir = "",
    [string]$Machine = "CPU_X_8",
    [string]$Cloud = "gcp-lightning-public-prod",
    [switch]$SkipUpload,
    [switch]$SkipCora,
    [switch]$SkipArxiv
)

$ErrorActionPreference = "Stop"

if (-not $env:LIGHTNING_USER_ID -or -not $env:LIGHTNING_API_KEY) {
    throw "Set LIGHTNING_USER_ID and LIGHTNING_API_KEY before launching Lightning jobs."
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $repoRoot ".venv_modal\Scripts\python.exe"
$launcher = Join-Path $repoRoot "scripts\lightning_production_benchmark.py"

if (-not $PackageModel) {
    $PackageModel = "jigsaw-production-benchmark-package-overlap-graphsage-$Version"
}
if (-not $PackageDir) {
    $PackageDir = "runs\lightning_production_benchmark_package_overlap_graphsage_$Version"
}

function Invoke-Launcher {
    param([string[]]$LauncherArgs)
    & $python $launcher @LauncherArgs
    if ($LASTEXITCODE -ne 0) {
        throw "lightning_production_benchmark.py failed: $($LauncherArgs -join ' ')"
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
        "--owner-kind", $OwnerKind,
        "--teamspace", $Teamspace,
        "--package-model", $PackageModel
    )
}

$common = @(
    "--owner", $Owner,
    "--owner-kind", $OwnerKind,
    "--teamspace", $Teamspace,
    "--package-model", $PackageModel,
    "--machine", $Machine,
    "--cloud", $Cloud,
    "--interruptible",
    "--workers", "1",
    "--parallel-mode", "task",
    "--queries", "50",
    "--target-sizes", "20,50,100",
    "--query-types", "all",
    "--seeds", "20260607,20260608",
    "--methods", "neural_component,mean_rrf_component",
    "--signature", "type_feat32",
    "--solver-timeout", "5"
)

if (-not $SkipCora) {
    $coraSpecs = @(
        "cora_overlap_best=/workspace/jigsaw_pkg/cache/models/cora-6_layer-model-graphsage_graphsage_final_loss_cora_seed7202_overlap_topk10_live20_best_fullcov.pth",
        "cora_overlap_final=/workspace/jigsaw_pkg/cache/models/cora-6_layer-model-graphsage_graphsage_final_loss_cora_seed7202_overlap_topk10_live20.pth"
    ) -join ";"
    Invoke-Launcher (@("launch-job") + $common + @(
        "--job-name", "jigsaw-cora-overlap-graphsage-bench-gcp-cpux8-$Version",
        "--result-model", "jigsaw-cora-overlap-graphsage-bench-$Version-results",
        "--dataset", "cora",
        "--budgets", "2,5,10,20",
        "--full-budget", "20",
        "--model-specs", $coraSpecs,
        "--output-prefix", "prod_cora_overlap_graphsage"
    ))
}

if (-not $SkipArxiv) {
    $arxivSpecs = @(
        "arxiv_overlap_best=/workspace/jigsaw_pkg/cache/models/arxiv-6_layer-model-graphsage_graphsage_final_loss_arxiv_seed7202_overlap_topk50_live64_best_fullcov.pth",
        "arxiv_overlap_final=/workspace/jigsaw_pkg/cache/models/arxiv-6_layer-model-graphsage_graphsage_final_loss_arxiv_seed7202_overlap_topk50_live64.pth"
    ) -join ";"
    Invoke-Launcher (@("launch-job") + $common + @(
        "--job-name", "jigsaw-arxiv-overlap-graphsage-bench-gcp-cpux8-$Version",
        "--result-model", "jigsaw-arxiv-overlap-graphsage-bench-$Version-results",
        "--dataset", "arxiv",
        "--budgets", "20,50,100,200",
        "--full-budget", "200",
        "--model-specs", $arxivSpecs,
        "--output-prefix", "prod_arxiv_overlap_graphsage"
    ))
}

Write-Host "[DONE] Overlap-trained GraphSAGE benchmark jobs submitted."
