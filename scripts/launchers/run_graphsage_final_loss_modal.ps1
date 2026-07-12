param(
    [ValidateSet("arxiv", "mag")]
    [string]$Dataset = "arxiv",

    [int]$TrainingSeed = 7101,

    [string]$Profile = "swathihrao28",

    [switch]$Resume,

    [switch]$Spawn,

    [int]$Epochs = 90,

    [int]$StepsPerEpoch = 100,

    [int]$CoverageTopK = -1,

    [int]$CoverageBucketSize = -1,

    [double]$CoverageCvarFraction = 0.25,

    [int]$MaxLivePositiveParts = 24,

    [int]$CacheRefreshSteps = -1,

    [int]$MaxTrainCoarseParts = -1,

    [string]$QueryTargetSizes = "",

    [int]$QuerySizeJitter = -1,

    [ValidateSet("graphsage", "rgcn")]
    [string]$EncoderKind = "graphsage",

    [double]$MomentumCacheDecay = 0.0,

    [ValidateSet("hard", "overlap", "overlap_union")]
    [string]$CoverageTargetMode = "hard",

    [string]$RunNameSuffix = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modal = Join-Path $repoRoot ".venv_modal\Scripts\modal.exe"
$logDir = Join-Path $repoRoot "runs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:MODAL_PROFILE = $Profile
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$datasetConfig = @{
    arxiv = @{
        RunName = "graphsage_final_loss_arxiv_seed${TrainingSeed}"
        CoverageTopK = "20"
        CoverageBucketSize = "10"
        MaxTrainCoarse = "50"
        CacheRefreshSteps = "20"
        CacheEncodeBatchSize = "1"
        CachePartitionGraphs = "0"
        HardNegativeSource = "graphs"
        ValidationTopKs = "20,50,100"
        ValidationQueries = "50"
        ValidationInterval = "5"
        QueryTargetSizes = "20,20,20,50,100"
        QuerySizeJitter = "5"
        Spawn = $false
    }
    mag = @{
        RunName = "graphsage_final_loss_mag_seed${TrainingSeed}"
        CoverageTopK = "20"
        CoverageBucketSize = "10"
        MaxTrainCoarse = "50"
        CacheRefreshSteps = "25"
        CacheEncodeBatchSize = "32"
        CachePartitionGraphs = "1"
        HardNegativeSource = "cache"
        ValidationTopKs = "20,50,100,200,500,1000"
        ValidationQueries = "50"
        ValidationInterval = "5"
        QueryTargetSizes = "20,20,20,50,100"
        QuerySizeJitter = "5"
        Spawn = $true
    }
}

$config = $datasetConfig[$Dataset]
$baseRunName = if ($EncoderKind -eq "rgcn") { $config.RunName -replace "^graphsage", "rgcn" } else { $config.RunName }
$runName = if ($RunNameSuffix) { "${baseRunName}_$RunNameSuffix" } else { $baseRunName }
$effectiveCoverageTopK = if ($CoverageTopK -gt 0) { "$CoverageTopK" } else { $config.CoverageTopK }
$effectiveCoverageBucketSize = if ($CoverageBucketSize -gt 0) { "$CoverageBucketSize" } else { $config.CoverageBucketSize }
$effectiveCacheRefreshSteps = if ($CacheRefreshSteps -gt 0) { "$CacheRefreshSteps" } else { $config.CacheRefreshSteps }
$effectiveMaxTrainCoarse = if ($MaxTrainCoarseParts -gt 0) { "$MaxTrainCoarseParts" } else { $config.MaxTrainCoarse }
$effectiveQueryTargetSizes = if ($QueryTargetSizes) { $QueryTargetSizes } else { $config.QueryTargetSizes }
$effectiveQuerySizeJitter = if ($QuerySizeJitter -ge 0) { "$QuerySizeJitter" } else { $config.QuerySizeJitter }

$arguments = @(
    "run", "--detach", "scripts/modal_train_graphsage.py",
    "--dataset", $Dataset,
    "--epochs", "$Epochs",
    "--steps-per-epoch", "$StepsPerEpoch",
    "--batch-size", "8",
    "--run-name", $runName,
    "--learning-rate", "5e-5",
    "--min-learning-rate", "5e-6",
    "--warmup-steps", "100",
    "--scheduler-type", "cosine",
    "--gamma-partition", "1.0",
    "--gamma-fine-partition", "0.0",
    "--coverage-temperature", "0.05",
    "--coverage-topk", $effectiveCoverageTopK,
    "--coverage-topk-bucket-size", $effectiveCoverageBucketSize,
    "--coverage-topk-weight", "0.35",
    "--coverage-topk-margin", "0.0",
    "--coverage-positive-aggregation", "cvar",
    "--coverage-cvar-fraction", "$CoverageCvarFraction",
    "--max-live-positive-parts", "$MaxLivePositiveParts",
    "--prob-k-hop", "0.55",
    "--prob-single-part", "0.10",
    "--prob-multi-coarse", "0.30",
    "--hard-negative-source", $config.HardNegativeSource,
    "--query-target-sizes", $effectiveQueryTargetSizes,
    "--query-size-jitter", $effectiveQuerySizeJitter,
    "--max-gpos-nodes", "2500",
    "--max-train-coarse-parts", $effectiveMaxTrainCoarse,
    "--cache-refresh-steps", $effectiveCacheRefreshSteps,
    "--cache-encode-batch-size", $config.CacheEncodeBatchSize,
    "--cache-partition-graphs", $config.CachePartitionGraphs,
    "--validation-queries", $config.ValidationQueries,
    "--validation-interval", $config.ValidationInterval,
    "--validation-seeds", "31415,27182",
    "--validation-topks", $config.ValidationTopKs,
    "--early-stopping-patience", "0",
    "--training-seed", "$TrainingSeed",
    "--encoder-kind", $EncoderKind,
    "--momentum-cache-decay", "$MomentumCacheDecay",
    "--coverage-target-mode", $CoverageTargetMode
)

if ($Spawn -or $config.Spawn) {
    $arguments += "--spawn"
}

if (-not $Resume) {
    $arguments += "--fresh"
}

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
    TrainingSeed = $TrainingSeed
    Profile = $Profile
    OptimizerSteps = $Epochs * $StepsPerEpoch
    Objective = "$EncoderKind + final FullCov loss"
    RunName = $runName
    EncoderKind = $EncoderKind
    MomentumCacheDecay = $MomentumCacheDecay
    CoverageTargetMode = $CoverageTargetMode
    CoverageTopK = $effectiveCoverageTopK
    CoverageBucketSize = $effectiveCoverageBucketSize
    CoverageCvarFraction = $CoverageCvarFraction
    MaxLivePositiveParts = $MaxLivePositiveParts
    CacheRefreshSteps = $effectiveCacheRefreshSteps
    MaxTrainCoarseParts = $effectiveMaxTrainCoarse
    QueryTargetSizes = $effectiveQueryTargetSizes
    QuerySizeJitter = $effectiveQuerySizeJitter
    Spawn = [bool]($Spawn -or $config.Spawn)
    ProcessId = $process.Id
    Stdout = $stdout
    Stderr = $stderr
} | Format-Table -AutoSize
