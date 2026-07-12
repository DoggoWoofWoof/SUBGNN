#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONUNBUFFERED=1
export JIGSAW_TORCH_THREADS="${JIGSAW_TORCH_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$JIGSAW_TORCH_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$JIGSAW_TORCH_THREADS}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

CACHE_ROOT="${CACHE_ROOT:-$PWD/cache}"
RESULT_ROOT="${RESULT_ROOT:-$PWD/runs/lightning_mag_benchmark_results}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/models}"
RUN_MODE="${RUN_MODE:-benchmark}"
DATASET="${DATASET:-mag}"
CASCADE_CACHE_DIR="${CASCADE_CACHE_DIR:-$CACHE_ROOT/overlap_cascade}"
if [[ -z "${HIERARCHY_PATH:-}" ]]; then
  if [[ "$DATASET" == "mag" ]]; then
    HIERARCHY_PATH="$CACHE_ROOT/mag_hierarchies_type_rel_2000_fine5_finecov_v1.pt"
  else
    DATASET_HIERARCHY_PATH="$CACHE_ROOT/${DATASET}_hierarchies_finecov_v1.pt"
    if [[ -f "$DATASET_HIERARCHY_PATH" ]]; then
      HIERARCHY_PATH="$DATASET_HIERARCHY_PATH"
    else
      HIERARCHY_PATH=""
    fi
  fi
fi
BEST_MODEL="${BEST_MODEL:-$MODEL_DIR/mag-6_layer-model-rgcn_rgcn_final_loss_mag_seed7202_overlap_topk50_live64_best_fullcov.pth}"
FINAL_MODEL="${FINAL_MODEL:-$CACHE_ROOT/mag_rgcn_final_loss_mag_seed7202_overlap_topk50_live64_checkpoint.pth}"
MODEL_SPECS="${MODEL_SPECS:-}"
if [[ -z "$MODEL_SPECS" && "$DATASET" == "mag" ]]; then
  MODEL_SPECS="mag_rgcn_best=$BEST_MODEL;mag_rgcn_final=$FINAL_MODEL"
fi

mkdir -p "$CACHE_ROOT" "$MODEL_DIR" "$RESULT_ROOT"

echo "[MACHINE] nproc=$(nproc)"
free -h || true
df -h "$PWD" || true
(
  while true; do
    echo "[RESOURCE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    free -h || true
    ps -eo pid,ppid,pcpu,pmem,rss,comm,args --sort=-rss | head -n 12 || true
    sleep "${RESOURCE_LOG_INTERVAL:-60}"
  done
) &
RESOURCE_PID=$!
UPLOAD_PID=""
trap 'kill "$RESOURCE_PID" >/dev/null 2>&1 || true; if [[ -n "$UPLOAD_PID" ]]; then kill "$UPLOAD_PID" >/dev/null 2>&1 || true; fi' EXIT

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements_lightning_rgcn.txt
python -m pip install -q "faiss-cpu>=1.7.4" "scipy"

if [[ "$RUN_MODE" != "query-cache" ]] && ! command -v glasgow_subgraph_solver >/dev/null 2>&1; then
  echo "[SETUP] Glasgow solver missing; building locally"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential g++ libboost-all-dev libgmp-dev
  fi
  rm -rf /tmp/glasgow-subgraph-solver
  git clone --depth 1 https://github.com/ciaranm/glasgow-subgraph-solver.git /tmp/glasgow-subgraph-solver
  find /tmp/glasgow-subgraph-solver -type f -name 'CMakeLists.txt' -exec sed -i 's/-march=native/-march=x86-64/g' {} +
  cmake -S /tmp/glasgow-subgraph-solver -B /tmp/glasgow-subgraph-solver/build -DCMAKE_CXX_FLAGS='-march=x86-64'
  cmake --build /tmp/glasgow-subgraph-solver/build -j"$(nproc)"
  cp /tmp/glasgow-subgraph-solver/build/glasgow_subgraph_solver /usr/local/bin/glasgow_subgraph_solver
  chmod +x /usr/local/bin/glasgow_subgraph_solver
fi

if [[ "$DATASET" == "mag" ]]; then
  test -f "$HIERARCHY_PATH" || { echo "[ERROR] Missing hierarchy: $HIERARCHY_PATH"; exit 2; }
fi
if [[ -n "$MODEL_SPECS" ]]; then
  IFS=';' read -ra MODEL_SPEC_ARRAY <<< "$MODEL_SPECS"
  for spec in "${MODEL_SPEC_ARRAY[@]}"; do
    [[ -z "$spec" ]] && continue
    model_path="${spec#*=}"
    [[ -z "$model_path" ]] && continue
    test -f "$model_path" || { echo "[ERROR] Missing model from MODEL_SPECS: $model_path"; exit 2; }
  done
fi

if [[ -n "${LIGHTNING_QUERY_CACHE_MODEL:-}" && "${RUN_MODE}" != "query-cache" ]]; then
  python - <<'PY'
import os
from pathlib import Path
from lightning_sdk.models import download_model

target = Path(os.environ.get("CASCADE_CACHE_DIR", "cache/overlap_cascade")).resolve()
target.mkdir(parents=True, exist_ok=True)
model = os.environ["LIGHTNING_QUERY_CACHE_MODEL"]
print(f"[LIGHTNING] Downloading query cache {model} -> {target}", flush=True)
download_model(model, str(target), progress_bar=True)
copied = 0
for path in list(target.rglob("*_queries.pt")):
    dst = target / path.name
    if path.resolve() == dst.resolve():
        continue
    dst.write_bytes(path.read_bytes())
    copied += 1
print(f"[LIGHTNING] Query cache files at root: {len(list(target.glob('*_queries.pt')))} (copied {copied})", flush=True)
PY
fi

echo "[BENCH] dataset=$DATASET"
echo "[BENCH] hierarchy=$HIERARCHY_PATH"
echo "[BENCH] models=$MODEL_SPECS"

if [[ "${LIGHTNING_RESUME_RESULTS:-1}" == "1" && -n "${LIGHTNING_RESULTS_MODEL:-}" && "$RUN_MODE" != "query-cache" ]]; then
  python - <<'PY' || true
import os
import shutil
from pathlib import Path
from lightning_sdk.models import download_model

def normalize_backslash_paths(root: Path) -> int:
    moved = 0
    for src in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if "\\" not in src.name:
            continue
        dst = src.parent.joinpath(*[part for part in src.name.split("\\") if part])
        if src == dst:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                for child in src.iterdir():
                    child_dst = dst / child.name
                    if child_dst.exists():
                        if child.is_dir():
                            shutil.rmtree(child_dst)
                        else:
                            child_dst.unlink()
                    shutil.move(str(child), str(child_dst))
                src.rmdir()
            else:
                shutil.move(str(src), str(dst))
        else:
            if dst.exists():
                # Keep whichever copy has more bytes; partial CSV uploads can race.
                if src.stat().st_size > dst.stat().st_size:
                    dst.unlink()
                    shutil.move(str(src), str(dst))
                else:
                    src.unlink()
            else:
                shutil.move(str(src), str(dst))
        moved += 1
    return moved

target = Path(os.environ.get("RESULT_ROOT", "runs/lightning_mag_benchmark_results")).resolve()
cache_target = Path(os.environ.get("CASCADE_CACHE_DIR", "cache/overlap_cascade")).resolve()
download_root = Path(os.environ.get("CACHE_ROOT", "cache")).resolve() / "lightning_results_resume_download"
model = os.environ.get("LIGHTNING_RESULTS_MODEL", "")
if model:
    print(f"[LIGHTNING] Trying previous results {model} -> {download_root}", flush=True)
    if download_root.exists():
        shutil.rmtree(download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    download_model(model, str(download_root), progress_bar=True)
    normalized = normalize_backslash_paths(download_root)
    target.mkdir(parents=True, exist_ok=True)
    cache_target.mkdir(parents=True, exist_ok=True)
    result_entries = 0
    for item in download_root.iterdir():
        if item.name in {"overlap_cascade", "derived_cache"}:
            continue
        dst = target / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
        result_entries += 1
    cache_entries = 0
    for cache_dir_name in ("overlap_cascade", "derived_cache"):
        cache_dir = download_root / cache_dir_name
        if not cache_dir.exists():
            continue
        for src in cache_dir.glob("*.pt"):
            shutil.copy2(src, cache_target / src.name)
            cache_entries += 1
    print(
        f"[LIGHTNING] Previous results downloaded: result_entries={result_entries} "
        f"derived_cache_files={cache_entries} normalized_paths={normalized}",
        flush=True,
    )
PY
fi

if [[ "${LIGHTNING_PERIODIC_UPLOAD:-1}" == "1" && -n "${LIGHTNING_RESULTS_MODEL:-}" && "$RUN_MODE" != "query-cache" ]]; then
  (
    sleep "${LIGHTNING_UPLOAD_INITIAL_DELAY:-600}"
    while true; do
      python - <<'PY' || true
import os
import shutil
from pathlib import Path
from lightning_sdk.models import upload_model

result_root = Path(os.environ.get("RESULT_ROOT", "runs/lightning_mag_benchmark_results")).resolve()
cache_root = Path(os.environ.get("CASCADE_CACHE_DIR", "cache/overlap_cascade")).resolve()
bundle = Path(os.environ.get("CACHE_ROOT", "cache")).resolve() / "lightning_results_upload_bundle"
name = os.environ.get("LIGHTNING_RESULTS_MODEL", "")
if name and result_root.exists():
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    for item in result_root.iterdir():
        dst = bundle / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    derived = bundle / "overlap_cascade"
    derived.mkdir(exist_ok=True)
    patterns = (
        "*_prepared_hierarchy.pt",
        "*_overlap_index.pt",
        "*_coarse_embeddings.pt",
        "*_signature_tokens.pt",
        "*_feature_label_tokens.pt",
        "*_coarse_mean_features.pt",
        "*_coarse_topo_features.pt",
    )
    copied = 0
    if cache_root.exists():
        for pattern in patterns:
            for src in cache_root.glob(pattern):
                shutil.copy2(src, derived / src.name)
                copied += 1
    print(f"[LIGHTNING] Periodic upload {bundle} -> {name} derived_cache_files={copied}", flush=True)
    upload_model(name, path=bundle, progress_bar=False)
    print("[LIGHTNING] Periodic upload complete", flush=True)
PY
      sleep "${LIGHTNING_UPLOAD_INTERVAL:-600}"
    done
  ) &
  UPLOAD_PID=$!
fi

if [[ "$RUN_MODE" == "query-cache" ]]; then
  python scripts/run_mag_benchmark_matrix_local.py \
    --dataset "$DATASET" \
    --queries "${QUERIES_PER_TYPE:-50}" \
    --target-sizes "${TARGET_SIZES:-20,50,100}" \
    --query-types "${QUERY_TYPES:-all}" \
    --seeds "${SEEDS:-20260607,20260608}" \
    --methods "" \
    --model-specs "$MODEL_SPECS" \
    --hierarchy-path "$HIERARCHY_PATH" \
    --budgets "${BUDGETS:-20,50,100,200,500,1000}" \
    --full-budget "${FULL_BUDGET:-2000}" \
    --signature "${SIGNATURE:-type_rel_feat32}" \
    --solver-timeout "${SOLVER_TIMEOUT:-5}" \
    --data-root "${DATA_ROOT:-$PWD/data}" \
    --cache-dir "$CASCADE_CACHE_DIR" \
    --output-dir "$RESULT_ROOT" \
    --output-prefix "${OUTPUT_PREFIX:-prod_${DATASET}_seed7202}" \
    --glasgow-bin "${GLASGOW_SOLVER_BIN:-glasgow_subgraph_solver}" \
    --workers 1 \
    --skip-existing

  if [[ -n "${LIGHTNING_QUERY_CACHE_MODEL:-}" ]]; then
    QUERY_UPLOAD_DIR="$CACHE_ROOT/query_cache_upload"
    rm -rf "$QUERY_UPLOAD_DIR"
    mkdir -p "$QUERY_UPLOAD_DIR"
    find "$CASCADE_CACHE_DIR" -maxdepth 1 -type f -name '*_queries.pt' -exec cp {} "$QUERY_UPLOAD_DIR/" \;
    ls -lh "$QUERY_UPLOAD_DIR"
    python - <<'PY'
import os
from pathlib import Path
from lightning_sdk.models import upload_model

source = Path(os.environ.get("CACHE_ROOT", "cache")) / "query_cache_upload"
model = os.environ["LIGHTNING_QUERY_CACHE_MODEL"]
print(f"[LIGHTNING] Uploading query cache {source} -> {model}", flush=True)
upload_model(model, path=source, progress_bar=True)
PY
  fi
  exit 0
fi

python scripts/run_mag_benchmark_matrix_local.py \
  --dataset "$DATASET" \
  --queries "${QUERIES_PER_TYPE:-50}" \
  --target-sizes "${TARGET_SIZES:-20,50,100}" \
  --query-types "${QUERY_TYPES:-all}" \
  --seeds "${SEEDS:-20260607,20260608}" \
  --methods "${METHODS:-neural_component,random_component,mean_feature_component,mean_rrf_component,filterall_component}" \
  --ablation-set "${ABLATION_SET:-none}" \
  --label-source "${LABEL_SOURCE:-feature}" \
  --model-specs "$MODEL_SPECS" \
  --hierarchy-path "$HIERARCHY_PATH" \
  --budgets "${BUDGETS:-20,50,100,200,500,1000}" \
  --full-budget "${FULL_BUDGET:-2000}" \
  --signature "${SIGNATURE:-type_rel_feat32}" \
  --solver-timeout "${SOLVER_TIMEOUT:-5}" \
  --data-root "${DATA_ROOT:-$PWD/data}" \
  --cache-dir "$CASCADE_CACHE_DIR" \
  --output-dir "$RESULT_ROOT" \
  --output-prefix "${OUTPUT_PREFIX:-prod_${DATASET}_seed7202}" \
  --glasgow-bin "${GLASGOW_SOLVER_BIN:-glasgow_subgraph_solver}" \
  --workers "${CASCADE_WORKERS:-4}" \
  --parallel-mode "${CASCADE_PARALLEL_MODE:-task}" \
  --max-component-diag-nodes "${MAX_COMPONENT_DIAG_NODES:-50000}" \
  --max-component-solver-components "${MAX_COMPONENT_SOLVER_COMPONENTS:-50}" \
  --max-eval-queries "${MAX_EVAL_QUERIES:-0}" \
  --skip-existing

mapfile -t FINAL_PER_QUERY_CSVS < <(
  find "$RESULT_ROOT/results" -maxdepth 1 -type f \
    -name '*_per_query.csv' ! -name '*_partial_per_query.csv' | sort
)
if [[ "${#FINAL_PER_QUERY_CSVS[@]}" -gt 0 ]]; then
  python scripts/summarize_production_benchmarks.py \
    "${FINAL_PER_QUERY_CSVS[@]}" \
    --output "$RESULT_ROOT/mag_rgcn_production_summary.csv" || true
else
  echo "[WARN] No final per-query CSVs found for summary" >&2
fi

if [[ "${LIGHTNING_UPLOAD_RESULTS:-1}" == "1" ]]; then
  python - <<'PY'
import os
import shutil
from pathlib import Path
from lightning_sdk.models import upload_model

result_root = Path(os.environ.get("RESULT_ROOT", str(Path.cwd() / "runs/lightning_mag_benchmark_results"))).resolve()
cache_root = Path(os.environ.get("CASCADE_CACHE_DIR", "cache/overlap_cascade")).resolve()
bundle = Path(os.environ.get("CACHE_ROOT", "cache")).resolve() / "lightning_results_upload_bundle"
name = os.environ.get("LIGHTNING_RESULTS_MODEL", "jigsaw-mag-rgcn-benchmark-results-latest")
if bundle.exists():
    shutil.rmtree(bundle)
bundle.mkdir(parents=True, exist_ok=True)
for item in result_root.iterdir():
    dst = bundle / item.name
    if item.is_dir():
        shutil.copytree(item, dst)
    else:
        shutil.copy2(item, dst)
derived = bundle / "overlap_cascade"
derived.mkdir(exist_ok=True)
patterns = (
    "*_prepared_hierarchy.pt",
    "*_overlap_index.pt",
    "*_coarse_embeddings.pt",
    "*_signature_tokens.pt",
    "*_feature_label_tokens.pt",
    "*_coarse_mean_features.pt",
    "*_coarse_topo_features.pt",
)
copied = 0
if cache_root.exists():
    for pattern in patterns:
        for src in cache_root.glob(pattern):
            shutil.copy2(src, derived / src.name)
            copied += 1
print(f"[LIGHTNING] Uploading benchmark results {bundle} -> {name} derived_cache_files={copied}", flush=True)
upload_model(name, path=bundle, progress_bar=True)
print("[LIGHTNING] Upload complete", flush=True)
PY
fi
