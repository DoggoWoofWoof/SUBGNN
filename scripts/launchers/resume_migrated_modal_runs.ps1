$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modal = Join-Path $repoRoot ".venv_modal\Scripts\modal.exe"
$logDir = Join-Path $repoRoot "runs\logs"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$currentProfile = (& $modal profile current).Trim()
if ($currentProfile -ne "pilgnnteam") {
    & $modal profile activate pilgnnteam
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

function Start-MigratedRun {
    param(
        [Parameter(Mandatory = $true)]
        [string] $RunName,
        [Parameter(Mandatory = $true)]
        [string[]] $ExtraArgs
    )

    $stdout = Join-Path $logDir "${RunName}_pilgnnteam_resume_${timestamp}.out.log"
    $stderr = Join-Path $logDir "${RunName}_pilgnnteam_resume_${timestamp}.err.log"
    $arguments = @(
        "run", "--detach", "scripts\train_jigsaw_model.py",
        "--dataset", "arxiv",
        "--epochs", "40",
        "--batch-size", "8",
        "--steps-per-epoch", "75",
        "--num-hierarchies", "1",
        "--run-name", $RunName,
        "--gamma-partition", "1.5",
        "--gamma-fine-partition", "0.0",
        "--coverage-temperature", "0.05",
        "--alpha", "0.15",
        "--prob-k-hop", "0.70",
        "--prob-single-part", "0.03",
        "--prob-multi-coarse", "0.22",
        "--max-gpos-nodes", "4000",
        "--max-train-coarse-parts", "100",
        "--cache-refresh-steps", "50",
        "--learning-rate", "0.00002",
        "--scheduler-type", "plateau",
        "--min-learning-rate", "0.000005",
        "--warmup-steps", "0",
        "--plateau-patience", "5",
        "--plateau-factor", "0.5"
    ) + $ExtraArgs

    $process = Start-Process `
        -FilePath $modal `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru

    [pscustomobject]@{
        RunName = $RunName
        ProcessId = $process.Id
        Stdout = $stdout
        Stderr = $stderr
    }
}

$runs = @(
    Start-MigratedRun `
        -RunName "coverage_v4_nodebeta_from_v2_e40" `
        -ExtraArgs @("--beta", "0.10")

    Start-MigratedRun `
        -RunName "coverage_v5_topkbarrier_from_v2_e40" `
        -ExtraArgs @(
            "--beta", "0.05",
            "--coverage-topk", "20",
            "--coverage-topk-weight", "0.35",
            "--coverage-topk-margin", "0.0"
        )
)

$runs | Format-Table -AutoSize
Write-Host "Both runs will resume from their matching checkpoints in pilgnnteam:jigsaw-cache-vol."
