param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cora", "citeseer", "pubmed", "physics", "flickr", "mag", "yelp")]
    [string]$Dataset,

    [int]$TrainingSeed = 7201,

    [switch]$Smoke,

    [switch]$TimingProbe,

    [string]$QueryTargetSizes = "",

    [int]$QuerySizeJitter = -1,

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
Remove-Item Env:\MODAL_TOKEN_ID -ErrorAction SilentlyContinue
Remove-Item Env:\MODAL_TOKEN_SECRET -ErrorAction SilentlyContinue

$datasetConfig = @{
    cora = @{ Coarse = 20; CoverageTopK = 2; BucketSize = 1; MaxTrainCoarse = 20; QueryTargetSizes = "20,20,20,50,100"; QuerySizeJitter = 5 }
    citeseer = @{ Coarse = 10; CoverageTopK = 1; BucketSize = 1; MaxTrainCoarse = 10; QueryTargetSizes = "20,20,20,50,100"; QuerySizeJitter = 5 }
    pubmed = @{ Coarse = 20; CoverageTopK = 2; BucketSize = 1; MaxTrainCoarse = 20; QueryTargetSizes = "20,20,20,50,100"; QuerySizeJitter = 5 }
    physics = @{ Coarse = 35; CoverageTopK = 4; BucketSize = 2; MaxTrainCoarse = 35; QueryTargetSizes = "20,20,20,50,100"; QuerySizeJitter = 5 }
    flickr = @{
        Coarse = 100
        CoverageTopK = 10
        BucketSize = 5
        MaxTrainCoarse = 50
        QueryTargetSizes = "20,20,20,50,100"
        QuerySizeJitter = 5
        ValidationTopKs = "20,50,100"
        CacheRefreshSteps = 25
        CacheEncodeBatchSize = 32
        CachePartitionGraphs = 1
        HardNegativeSource = "cache"
    }
    mag = @{
        Coarse = 2000
        CoverageTopK = 20
        BucketSize = 10
        MaxTrainCoarse = 50
        QueryTargetSizes = "20,20,20,50,100"
        QuerySizeJitter = 5
        ValidationTopKs = "20,50,100,200,500,1000"
        CacheRefreshSteps = 25
        CacheEncodeBatchSize = 32
        CachePartitionGraphs = 1
        HardNegativeSource = "cache"
    }
    yelp = @{
        Coarse = 700
        CoverageTopK = 5
        BucketSize = 2
        MaxTrainCoarse = 20
        QueryTargetSizes = "20,20,20,50,100"
        QuerySizeJitter = 5
        ValidationTopKs = "20,50,100,200,500"
        CacheRefreshSteps = 25
        CacheEncodeBatchSize = 32
        CachePartitionGraphs = 1
        HardNegativeSource = "cache"
    }
}

$config = $datasetConfig[$Dataset]
$epochs = if ($Smoke) { 2 } elseif ($TimingProbe) { 1 } else { 90 }
$stepsPerEpoch = if ($Smoke) { 10 } elseif ($TimingProbe) { 100 } else { 100 }
$validationQueries = if ($Smoke) { 10 } elseif ($TimingProbe) { 0 } else { 50 }
$validationInterval = if ($Smoke) { 1 } elseif ($TimingProbe) { 1 } else { 5 }
$cacheRefreshSteps = if ($config.CacheRefreshSteps) { $config.CacheRefreshSteps } else { 20 }
$cacheEncodeBatchSize = if ($config.CacheEncodeBatchSize) { $config.CacheEncodeBatchSize } else { 1 }
$cachePartitionGraphs = if ($config.CachePartitionGraphs) { $config.CachePartitionGraphs } else { 0 }
$hardNegativeSource = if ($config.HardNegativeSource) { $config.HardNegativeSource } else { "graphs" }
$effectiveQueryTargetSizes = if ($QueryTargetSizes) { $QueryTargetSizes } else { $config.QueryTargetSizes }
$effectiveQuerySizeJitter = if ($QuerySizeJitter -ge 0) { $QuerySizeJitter } else { $config.QuerySizeJitter }
$mode = if ($Smoke) { "smoke" } elseif ($TimingProbe) { "speedprobe" } else { "final" }
$runName = "coverage_cross_dataset_${Dataset}_${mode}_seed${TrainingSeed}"

$arguments = @(
    "run", "--detach", "scripts/train_jigsaw_model.py",
    "--dataset", $Dataset,
    "--epochs", "$epochs",
    "--steps-per-epoch", "$stepsPerEpoch",
    "--batch-size", "8",
    "--run-name", $runName,
    "--learning-rate", "5e-5",
    "--min-learning-rate", "5e-6",
    "--warmup-steps", "100",
    "--scheduler-type", "cosine",
    "--gamma-partition", "1.0",
    "--gamma-fine-partition", "0.0",
    "--coverage-temperature", "0.05",
    "--coverage-topk", "$($config.CoverageTopK)",
    "--coverage-topk-bucket-size", "$($config.BucketSize)",
    "--coverage-topk-weight", "0.35",
    "--coverage-topk-margin", "0.0",
    "--coverage-positive-aggregation", "cvar",
    "--coverage-cvar-fraction", "0.25",
    "--max-live-positive-parts", "24",
    "--prob-k-hop", "0.55",
    "--prob-single-part", "0.10",
    "--prob-multi-coarse", "0.30",
    "--hard-negative-source", "$hardNegativeSource",
    "--query-target-sizes", "$effectiveQueryTargetSizes",
    "--query-size-jitter", "$effectiveQuerySizeJitter",
    "--max-gpos-nodes", "2500",
    "--max-train-coarse-parts", "$($config.MaxTrainCoarse)",
    "--cache-refresh-steps", "$cacheRefreshSteps",
    "--cache-encode-batch-size", "$cacheEncodeBatchSize",
    "--cache-partition-graphs", "$cachePartitionGraphs",
    "--validation-queries", "$validationQueries",
    "--validation-interval", "$validationInterval",
    "--validation-seeds", "31415,27182",
    "--validation-topks", $(if ($config.ValidationTopKs) { $config.ValidationTopKs } else { "20,50,100" }),
    "--early-stopping-patience", "0",
    "--training-seed", "$TrainingSeed",
    "--spawn",
    "--fresh"
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logDir "${runName}_${timestamp}.out.log"
$stderr = Join-Path $logDir "${runName}_${timestamp}.err.log"
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
    CoarsePartitions = $config.Coarse
    CoverageTopK = $config.CoverageTopK
    CoverageBucketSize = $config.BucketSize
    CacheRefreshSteps = $cacheRefreshSteps
    CacheEncodeBatchSize = $cacheEncodeBatchSize
    CachePartitionGraphs = $cachePartitionGraphs
    HardNegativeSource = $hardNegativeSource
    QueryTargetSizes = $effectiveQueryTargetSizes
    QuerySizeJitter = $effectiveQuerySizeJitter
    TrainingSeed = $TrainingSeed
    Smoke = [bool]$Smoke
    TimingProbe = [bool]$TimingProbe
    OptimizerSteps = $epochs * $stepsPerEpoch
    Profile = $Profile
    RunName = $runName
    ProcessId = $process.Id
    Stdout = $stdout
    Stderr = $stderr
} | Format-Table -AutoSize
