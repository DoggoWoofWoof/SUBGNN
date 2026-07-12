#!/usr/bin/env bash
# Linux/Lightning launcher for the Cora GNN-PE feasibility spike.
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONUNBUFFERED=1
export DATASET="${DATASET:-cora}"
export DATA_ROOT="${DATA_ROOT:-data}"
export SPIKE_OUTPUT="${SPIKE_OUTPUT:-runs/gnnpe_spike/${DATASET}}"
export GNNPE_REPO="${GNNPE_REPO:-external/GNN-PE}"
export QUERY_TYPES="${QUERY_TYPES:-small_bfs,k_hop,single,multi_fine,multi_coarse,random_walk}"
export LABEL_SOURCE="${LABEL_SOURCE:-class}"
export TARGET_SIZES="${TARGET_SIZES:-8,20,50,100}"
export QUERIES_PER_CELL="${QUERIES_PER_CELL:-1}"
export ANSWER_LIMIT="${ANSWER_LIMIT:-1}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
export GNNPE_PARTITIONS="${GNNPE_PARTITIONS:-20}"
export GNNPE_PATH_LENGTH="${GNNPE_PATH_LENGTH:-2}"
export GNNPE_EMBED_DIM="${GNNPE_EMBED_DIM:-2}"
export HIERARCHY_PATH="${HIERARCHY_PATH:-runs/overlap_models/cora/cora_hierarchies_finecov_v1.pt}"

mkdir -p "$SPIKE_OUTPUT"
exec > >(tee -a "$SPIKE_OUTPUT/launcher.log") 2>&1

upload_results_on_exit() {
  local rc=$?
  set +e
  echo "[EXIT] run_gnnpe_feasibility_spike.sh rc=$rc"
  if [[ "${LIGHTNING_UPLOAD_RESULTS:-1}" == "1" && -n "${LIGHTNING_RESULTS_MODEL:-}" && -d "$SPIKE_OUTPUT" ]]; then
    echo "[LIGHTNING] Trimming spike output before upload"
    rm -rf \
      "$SPIKE_OUTPUT/gnnpe_build" \
      "$SPIKE_OUTPUT/gnn-pe" \
      "$SPIKE_OUTPUT/data_graph.gpickle"
    find "$SPIKE_OUTPUT" -maxdepth 1 -type f \( \
      -name '*.idx' -o \
      -name '*.index' -o \
      -name '*.dat' -o \
      -name '*.bin' -o \
      -name '*.npy' -o \
      -name '*.npz' \
    \) -delete
    python - <<'PY'
import os
from lightning_sdk.models import upload_model
path = os.environ["SPIKE_OUTPUT"]
name = os.environ["LIGHTNING_RESULTS_MODEL"]
print(f"[LIGHTNING] Uploading GNN-PE spike results {path} -> {name}", flush=True)
try:
    upload_model(name, path=path, progress_bar=False)
except Exception as exc:
    print(f"[LIGHTNING] Upload failed: {exc!r}", flush=True)
else:
    print("[LIGHTNING] Upload complete", flush=True)
PY
  fi
  exit "$rc"
}
trap upload_results_on_exit EXIT

echo "[STEP] Installing system and Python dependencies"
if command -v apt-get >/dev/null 2>&1; then
  echo "[STEP] Installing build dependencies"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential git g++-12 || \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential git g++
fi

if command -v g++-12 >/dev/null 2>&1; then
  export CXX=g++-12
  export CC=gcc-12
fi
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements_lightning_rgcn.txt
python -m pip install -q faiss-cpu
python -m pip install -q "cmake>=3.28"
echo "[INFO] Python=$(python --version 2>&1)"
echo "[INFO] CMake=$(cmake --version 2>&1 | head -n 1 || true)"
echo "[INFO] CXX=$(${CXX:-g++} --version 2>&1 | head -n 1 || true)"

if [[ ! -d "$GNNPE_REPO/GNN-PE" ]]; then
  echo "[STEP] Cloning GNN-PE"
  mkdir -p "$(dirname "$GNNPE_REPO")"
  git clone --depth 1 https://github.com/JamesWhiteSnow/GNN-PE.git "$GNNPE_REPO"
fi

echo "[STEP] Exporting Cora workload to GNN-PE formats"
python scripts/analysis/gnnpe_feasibility_spike.py \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --output "$SPIKE_OUTPUT" \
  --hierarchy "$HIERARCHY_PATH" \
  --gnnpe-repo "$GNNPE_REPO" \
  --query-types "$QUERY_TYPES" \
  --label-source "$LABEL_SOURCE" \
  --target-sizes "$TARGET_SIZES" \
  --queries-per-cell "$QUERIES_PER_CELL" \
  --partition-num "$GNNPE_PARTITIONS" \
  --path-length "$GNNPE_PATH_LENGTH" \
  --embedding-dim "$GNNPE_EMBED_DIM" \
  --answer-limit "$ANSWER_LIMIT" \
  --timeout-seconds "$TIMEOUT_SECONDS"

echo "[STEP] Running GNN-PE spike"
bash "$SPIKE_OUTPUT/run_gnnpe_spike.sh"
