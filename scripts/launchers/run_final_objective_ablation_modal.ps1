param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("control", "cvar", "topk", "final")]
    [string]$Variant,

    [int]$TrainingSeed = 7101,

    [switch]$Resume,

    [string]$Profile = "pilgnnteam"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modal = Join-Path $repoRoot ".venv_modal\Scripts\modal.exe"
$logDir = Join-Path $repoRoot "runs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:MODAL_PROFILE = $Profile

$variantConfig = @{
    control = @{
        Aggregation = "mean"
        TopKWeight = "0.0"
        LivePositives = "0"
    }
    cvar = @{
        Aggregation = "cvar"
        TopKWeight = "0.0"
        LivePositives = "0"
    }
    topk = @{
        Aggregation = "cvar"
        TopKWeight = "0.35"
        LivePositives = "0"
    }
    final = @{
        Aggregation = "cvar"
        TopKWeight = "0.35"
        LivePositives = "24"
    }
}

$config = $variantConfig[$Variant]
$runName = "coverage_final_ablation_${Variant}_seed${TrainingSeed}"
$arguments = @(
    "run", "--detach", "scripts/train_jigsaw_model.py",
    "--dataset", "arxiv",
    "--epochs", "90",
    "--steps-per-epoch", "100",
    "--batch-size", "8",
    "--run-name", $runName,
    "--learning-rate", "5e-5",
    "--min-learning-rate", "5e-6",
    "--warmup-steps", "100",
    "--scheduler-type", "cosine",
    "--gamma-partition", "1.0",
    "--gamma-fine-partition", "0.0",
    "--coverage-temperature", "0.05",
    "--coverage-topk", "20",
    "--coverage-topk-weight", $config.TopKWeight,
    "--coverage-topk-margin", "0.0",
    "--coverage-positive-aggregation", $config.Aggregation,
    "--coverage-cvar-fraction", "0.25",
    "--max-live-positive-parts", $config.LivePositives,
    "--prob-k-hop", "0.55",
    "--prob-single-part", "0.10",
    "--prob-multi-coarse", "0.30",
    "--max-gpos-nodes", "2500",
    "--max-train-coarse-parts", "50",
    "--cache-refresh-steps", "20",
    "--validation-queries", "50",
    "--validation-interval", "5",
    "--validation-seeds", "31415,27182",
    "--early-stopping-patience", "0",
    "--training-seed", $TrainingSeed
)

if (-not $Resume) {
    $arguments += "--fresh"
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
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
    Variant = $Variant
    TrainingSeed = $TrainingSeed
    Resume = [bool]$Resume
    Profile = $Profile
    OptimizerSteps = 9000
    RunName = $runName
    ProcessId = $process.Id
    Stdout = $stdout
    Stderr = $stderr
} | Format-Table -AutoSize
