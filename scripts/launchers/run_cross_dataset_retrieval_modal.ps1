param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cora", "citeseer", "pubmed", "physics", "flickr", "arxiv", "mag", "yelp")]
    [string]$Dataset,

    [int]$Queries = 100,

    [string]$TargetSizes = "20,50,100",

    [int[]]$Seeds = @(20260607, 20260608),

    [string]$ModelSpecs = "",

    [string]$HierarchyPath = "",

    [string]$Profile = "deepalimohapatra1973",

    [switch]$IncludeKSweep,

    [switch]$IncludeOverlapNodeCoverage,

    [switch]$OverlapDiagnosticsOnly,

    [switch]$IncludeSignaturePruning,

    [string]$Tag = ""
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
    if ($Dataset -eq "arxiv") {
        $ModelSpecs = "arxiv_v7_continue_best=/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth"
    } else {
        $ModelSpecs = "${Dataset}_final_best=/cache/models/${Dataset}-6_layer-model-jigsaw_coverage_cross_dataset_${Dataset}_final_seed7201_best_fullcov.pth"
    }
}

$sizeTag = $TargetSizes.Replace(",", "_")
$launched = foreach ($seed in $Seeds) {
    $tagPart = if ($Tag) { "_${Tag}" } else { "" }
    $outputTag = "retrieval_${Dataset}_khop${tagPart}_q${Queries}_sizes${sizeTag}_seed${seed}"
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $logDir "${outputTag}_${timestamp}.out.log"
    $stderr = Join-Path $logDir "${outputTag}_${timestamp}.err.log"
    $arguments = @(
        "run", "--detach", "modal_retrieval_benchmark.py",
        "--dataset", $Dataset,
        "--model-specs", $ModelSpecs,
        "--output-tag", $outputTag,
        "--queries", "$Queries",
        "--target-sizes", $TargetSizes,
        "--seed", "$seed"
    )
    if ($HierarchyPath) {
        $arguments += @("--hierarchy-path", $HierarchyPath)
    }
    if ($IncludeKSweep) {
        $arguments += @("--include-k-sweep")
    }
    if ($IncludeOverlapNodeCoverage) {
        $arguments += @("--include-overlap-node-coverage")
    }
    if ($OverlapDiagnosticsOnly) {
        $arguments += @("--overlap-diagnostics-only")
    }
    if ($IncludeSignaturePruning) {
        $arguments += @("--include-signature-pruning")
    }

    $process = Start-Process `
        -FilePath $modal `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru

    [pscustomobject]@{
        Dataset = $Dataset
        Seed = $seed
        QueriesPerSize = $Queries
        TargetSizes = $TargetSizes
        Profile = $Profile
        ModelSpecs = $ModelSpecs
        ProcessId = $process.Id
        Stdout = $stdout
        Stderr = $stderr
    }
}

$launched | Format-Table -AutoSize
