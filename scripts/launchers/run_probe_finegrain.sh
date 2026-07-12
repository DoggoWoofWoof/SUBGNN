#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
export JIGSAW_TORCH_THREADS="${JIGSAW_TORCH_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$JIGSAW_TORCH_THREADS}"

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements_lightning_rgcn.txt
python -m pip install -q "faiss-cpu>=1.7.4" scipy

CACHE_ROOT="${CACHE_ROOT:-$PWD}"
HIER="${HIERARCHY_PATH:-$CACHE_ROOT/mag_hierarchies_type_rel_2000_fine5_finecov_v1.pt}"
if [[ ! -f "$HIER" ]]; then HIER="$(find "$PWD" -name 'mag_hierarchies_*finecov_v1.pt' | head -1)"; fi
MODEL="${MODEL_PATH:-}"
if [[ -z "$MODEL" ]]; then MODEL="$(find "$PWD" -path '*models*' -name '*rgcn*best_fullcov.pth' | head -1)"; fi
echo "[probe] HIER=$HIER"
echo "[probe] MODEL=$MODEL"
test -f "$HIER" || { echo "[ERR] hierarchy missing"; exit 2; }
test -f "$MODEL" || { echo "[ERR] model missing"; exit 2; }

OUTDIR="$PWD/runs/probe_finegrain"
mkdir -p "$OUTDIR"
python scripts/probe_finegrain_expansion.py \
  --dataset "${DATASET:-mag}" --data-root "$PWD/data" \
  --hierarchy-path "$HIER" --model "$MODEL" \
  --cache-dir "$PWD/cache/probe" \
  --seeds "${SEEDS:-20260607,20260608}" \
  --sizes "${TARGET_SIZES:-20,50,100}" \
  --queries "${QUERIES_PER_TYPE:-50}" \
  --query-types "${QUERY_TYPES:-single,multi_fine,multi_coarse,degree_k_hop,k_hop,random_walk}" \
  --stitch-budget "${STITCH_BUDGET:-200}" --stitch-pool "${STITCH_POOL:-400}" \
  --output "$OUTDIR/probe_finegrain_per_query.csv"

if [[ -n "${LIGHTNING_RESULTS_MODEL:-}" ]]; then
  python - <<'PY'
import os
from pathlib import Path
from lightning_sdk.models import upload_model
name=os.environ["LIGHTNING_RESULTS_MODEL"]
src=Path(os.getcwd())/"runs/probe_finegrain"
print(f"[probe] uploading {src} -> {name}", flush=True)
upload_model(name, path=str(src), progress_bar=True)
print("[probe] upload done", flush=True)
PY
fi
echo "[probe] DONE"
