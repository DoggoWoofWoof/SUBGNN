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
$stdout = Join-Path $logDir "retrieval_arxiv_khop_v2_v4_v5_q30_seed42_${timestamp}.out.log"
$stderr = Join-Path $logDir "retrieval_arxiv_khop_v2_v4_v5_q30_seed42_${timestamp}.err.log"

$process = Start-Process `
    -FilePath $modal `
    -ArgumentList @("run", "--detach", "modal_retrieval_benchmark.py") `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    ProcessId = $process.Id
    Stdout = $stdout
    Stderr = $stderr
} | Format-Table -AutoSize
