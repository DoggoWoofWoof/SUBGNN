param(
    [string]$Profile = "pilgnnteam",
    [string]$Volume = "jigsaw-cache-vol",
    [switch]$IncludeCleanV7
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modal = Join-Path $repoRoot ".venv_modal\Scripts\modal.exe"
$python = Join-Path $repoRoot ".venv_modal\Scripts\python.exe"
$migrationRoot = Join-Path $repoRoot "runs\modal_migration\$Profile\$Volume"
$modelMirror = Join-Path $repoRoot "models"

New-Item -ItemType Directory -Force -Path (Join-Path $migrationRoot "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $migrationRoot "logs") | Out-Null
New-Item -ItemType Directory -Force -Path $modelMirror | Out-Null

$env:MODAL_PROFILE = $Profile
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$files = @(
    "models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth",
    "models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6.pth",
    "arxiv_coverage_v7_continue_from_v6_checkpoint.pth",
    "logs/train_arxiv_coverage_v7_continue_from_v6.log"
)

if ($IncludeCleanV7) {
    $files += @(
        "models/arxiv-6_layer-model-jigsaw_coverage_v7_clean_seed7002_best_fullcov.pth",
        "models/arxiv-6_layer-model-jigsaw_coverage_v7_clean_seed7002.pth",
        "arxiv_coverage_v7_clean_seed7002_checkpoint.pth",
        "logs/train_arxiv_coverage_v7_clean_seed7002.log"
    )
}

foreach ($remote in $files) {
    $local = Join-Path $migrationRoot ($remote -replace "/", "\")
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $local) | Out-Null
    Write-Host "[MODAL] $Profile/${Volume}:$remote -> $local"
    & $modal volume get --force $Volume $remote $local
    if ($LASTEXITCODE -ne 0) {
        throw "modal volume get failed for $remote"
    }
}

$mirrorFiles = @(
    "arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth",
    "arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6.pth"
)
if ($IncludeCleanV7) {
    $mirrorFiles += @(
        "arxiv-6_layer-model-jigsaw_coverage_v7_clean_seed7002_best_fullcov.pth",
        "arxiv-6_layer-model-jigsaw_coverage_v7_clean_seed7002.pth"
    )
}

foreach ($name in $mirrorFiles) {
    $src = Join-Path $migrationRoot "models\$name"
    $dst = Join-Path $modelMirror $name
    Copy-Item -LiteralPath $src -Destination $dst -Force
    Write-Host "[LOCAL] mirrored $dst"
}

$manifest = Join-Path $migrationRoot "FETCHED_ARXIV_V7_MODELS.txt"
@(
    "profile=$Profile",
    "volume=$Volume",
    "fetched_at=$(Get-Date -Format o)",
    "source_app=https://modal.com/apps/pilgnnteam/main/ap-TRC5vfOW4krsJ2rbTsmHvR",
    "continuation_best=models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth",
    "continuation_final=models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6.pth"
) | Set-Content -Path $manifest -Encoding UTF8

$validateList = $mirrorFiles | ForEach-Object { Join-Path $modelMirror $_ }
$validateJson = ConvertTo-Json $validateList -Compress
@"
import json
import torch
from pathlib import Path

for raw in json.loads(r'''$validateJson'''):
    path = Path(raw)
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, (dict, torch.nn.Module)):
        raise TypeError(f"unexpected checkpoint type for {path}: {type(obj)!r}")
    print(f"[VALIDATE] ok {path} bytes={path.stat().st_size}")
"@ | & $python -

Write-Host "[DONE] Arxiv v7 models fetched, mirrored, and validated."
