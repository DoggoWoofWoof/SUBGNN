param(
  [string]$Profile = "crmohapatra",
  [string]$ArxivModel = "/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth",
  [string]$MagModel = "/cache/models/mag-6_layer-model-jigsaw_coverage_cross_dataset_mag_final_seed7201.pth",
  [string]$CoraModel = "/app/models/cora-6_layer-model-jigsaw.pth",
  [int]$QueriesPerType = 50,
  [string]$QueryTypes = "all",
  [string]$Seeds = "20260607,20260608",
  [string]$AblationSet = "full",
  [string]$Datasets = "cora,arxiv,mag",
  [string]$Methods = "all",
  [int]$LaunchDelaySec = 2
)

$ErrorActionPreference = "Stop"
$env:MODAL_PROFILE = $Profile
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force -Path "runs\logs" | Out-Null
$RootDir = (Get-Location).Path
$ModalExe = Join-Path $RootDir ".venv_modal\Scripts\modal.exe"
$LogDir = Join-Path $RootDir "runs\logs"

function Launch-Cascade {
  param(
    [string]$Dataset,
    [string]$Tag,
    [string]$ModelSpec,
    [string]$HierarchyPath,
    [string]$Budgets,
    [string]$Method,
    [string]$Signature,
    [string]$TargetSizes,
    [int]$Seed,
    [bool]$ComponentSolve = $true,
    [bool]$PruneLabels = $true,
    [bool]$UseOverlap = $true
  )
  $args = @(
    "run", "--detach", "modal_benchmark_glasgow.py",
    "--overlap-cascade",
    "--dataset", $Dataset,
    "--queries", "$QueriesPerType",
    "--target-sizes", $TargetSizes,
    "--cascade-query-types", $QueryTypes,
    "--seed", "$Seed",
    "--model-specs", $ModelSpec,
    "--output-tag", $Tag,
    "--cascade-budgets", $Budgets,
    "--cascade-method", $Method,
    "--cascade-signature", $Signature,
    "--solver-timeout", "5",
    "--max-component-diag-nodes", "50000",
    "--max-component-solver-components", "50"
  )
  if ($HierarchyPath) { $args += @("--hierarchy-path", $HierarchyPath) }
  if ($PruneLabels) { $args += "--prune-query-labels" }
  if ($ComponentSolve) { $args += "--component-solve" }
  if (-not $UseOverlap) { $args += "--no-overlap" }

  $out = Join-Path $LogDir "$Tag.launch.out.log"
  $err = Join-Path $LogDir "$Tag.launch.err.log"
  Start-Process -FilePath $ModalExe -ArgumentList $args -WorkingDirectory $RootDir -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden
  Write-Host "launched $Tag"
  Start-Sleep -Seconds $LaunchDelaySec
}

function Launch-Dataset {
  param(
    [string]$Dataset,
    [string]$ModelSpec,
    [string]$HierarchyPath,
    [string]$Budgets,
    [string]$FullBudget,
    [string]$Signature,
    [string]$TargetSizes,
    [int]$Seed
  )
  $prefix = "prod_${Dataset}_s${Seed}_q${QueriesPerType}_types_${QueryTypes}_sizes20_50_100"
  $methodSet = @{}
  $Methods.Split(",") | ForEach-Object { $methodSet[$_.Trim().ToLowerInvariant()] = $true }
  $allMethods = $methodSet.ContainsKey("all")

  if ($allMethods -or $methodSet.ContainsKey("fullgraph")) {
    Launch-Cascade $Dataset "${prefix}_fullgraph_b${FullBudget}" $ModelSpec $HierarchyPath $FullBudget "all" "none" $TargetSizes $Seed $false $false $false
  }
  if ($allMethods -or $methodSet.ContainsKey("filterall_component")) {
    Launch-Cascade $Dataset "${prefix}_filterall_component_b${FullBudget}" $ModelSpec $HierarchyPath $FullBudget "all" $Signature $TargetSizes $Seed $true $true $true
  }
  if ($allMethods -or $methodSet.ContainsKey("neural_component")) {
    Launch-Cascade $Dataset "${prefix}_neural_component_b$($Budgets.Replace(',', '_'))" $ModelSpec $HierarchyPath $Budgets "hybrid" $Signature $TargetSizes $Seed $true $true $true
  }
  if ($allMethods -or $methodSet.ContainsKey("random_component")) {
    Launch-Cascade $Dataset "${prefix}_random_component_b$($Budgets.Replace(',', '_'))" $ModelSpec $HierarchyPath $Budgets "random" $Signature $TargetSizes $Seed $true $true $true
  }
  if ($allMethods -or $methodSet.ContainsKey("mean_feature_component")) {
    Launch-Cascade $Dataset "${prefix}_mean_feature_component_b$($Budgets.Replace(',', '_'))" $ModelSpec $HierarchyPath $Budgets "mean_feature" $Signature $TargetSizes $Seed $true $true $true
  }
  if ($allMethods -or $methodSet.ContainsKey("mean_rrf_component")) {
    Launch-Cascade $Dataset "${prefix}_mean_rrf_component_b$($Budgets.Replace(',', '_'))" $ModelSpec $HierarchyPath $Budgets "coarse_mean_rrf" $Signature $TargetSizes $Seed $true $true $true
  }
  if ($allMethods -or $methodSet.ContainsKey("topo_feature_component")) {
    Launch-Cascade $Dataset "${prefix}_topo_feature_component_b$($Budgets.Replace(',', '_'))" $ModelSpec $HierarchyPath $Budgets "topo_feature" $Signature $TargetSizes $Seed $true $true $true
  }

  if ($AblationSet -eq "full") {
    if ($allMethods -or $methodSet.ContainsKey("neural_no_component")) {
      Launch-Cascade $Dataset "${prefix}_neural_no_component_b$($Budgets.Replace(',', '_'))" $ModelSpec $HierarchyPath $Budgets "hybrid" $Signature $TargetSizes $Seed $false $true $true
    }
    if ($allMethods -or $methodSet.ContainsKey("neural_no_overlap")) {
      Launch-Cascade $Dataset "${prefix}_neural_no_overlap_b$($Budgets.Replace(',', '_'))" $ModelSpec $HierarchyPath $Budgets "hybrid" $Signature $TargetSizes $Seed $true $true $false
    }
    if ($allMethods -or $methodSet.ContainsKey("neural_no_signature")) {
      Launch-Cascade $Dataset "${prefix}_neural_no_signature_b$($Budgets.Replace(',', '_'))" $ModelSpec $HierarchyPath $Budgets "hybrid" "none" $TargetSizes $Seed $true $true $true
    }
    if ($allMethods -or $methodSet.ContainsKey("neural_no_exact_label")) {
      Launch-Cascade $Dataset "${prefix}_neural_no_exact_label_b$($Budgets.Replace(',', '_'))" $ModelSpec $HierarchyPath $Budgets "hybrid" $Signature $TargetSizes $Seed $true $false $true
    }
  }
}

$seedList = $Seeds.Split(",") | ForEach-Object { [int]$_.Trim() }
$datasetSet = @{}
$Datasets.Split(",") | ForEach-Object { $datasetSet[$_.Trim().ToLowerInvariant()] = $true }
foreach ($seed in $seedList) {
  if ($datasetSet.ContainsKey("cora")) {
    Launch-Dataset "cora" "cora=$CoraModel" "" "2,5,10,20" "20" "type_feat32" "20,50,100" $seed
  }
  if ($datasetSet.ContainsKey("arxiv")) {
    Launch-Dataset "arxiv" "arxiv=$ArxivModel" "" "20,50,100,200" "200" "type_feat32" "20,50,100" $seed
  }
  if ($datasetSet.ContainsKey("mag")) {
    Launch-Dataset "mag" "mag=$MagModel" "/cache/mag_hierarchies_type_rel_2000_fine5_finecov_v1.pt" "20,50,100,200,500,1000" "2000" "type_rel_feat32" "20,50,100" $seed
  }
}

Write-Host "submitted matrix profile=$Profile datasets=$Datasets seeds=$Seeds queries_per_type=$QueriesPerType query_types=$QueryTypes methods=$Methods ablations=$AblationSet"
