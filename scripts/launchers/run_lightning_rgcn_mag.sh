#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export JIGSAW_TORCH_THREADS="${JIGSAW_TORCH_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$JIGSAW_TORCH_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$JIGSAW_TORCH_THREADS}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export JIGSAW_SKIP_MODAL_COMMIT="${JIGSAW_SKIP_MODAL_COMMIT:-1}"
export JIGSAW_UPLOAD_CHECKPOINTS="${JIGSAW_UPLOAD_CHECKPOINTS:-1}"

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements_lightning_rgcn.txt

python scripts/train_final_loss_local.py \
  --dataset mag \
  --epochs "${EPOCHS:-90}" \
  --steps-per-epoch "${STEPS_PER_EPOCH:-100}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --training-seed "${TRAINING_SEED:-7202}" \
  --cache-root "${CACHE_ROOT:-$PWD/cache}" \
  --encoder-kind rgcn \
  --momentum-cache-decay "${MOMENTUM_CACHE_DECAY:-0.99}" \
  --coverage-target-mode "${COVERAGE_TARGET_MODE:-overlap}" \
  --coverage-topk "${COVERAGE_TOPK:-50}" \
  --coverage-cvar-fraction "${COVERAGE_CVAR_FRACTION:-0.5}" \
  --max-live-positive-parts "${MAX_LIVE_POSITIVE_PARTS:-64}" \
  --max-train-coarse-parts "${MAX_TRAIN_COARSE_PARTS:-80}" \
  --cache-refresh-steps "${CACHE_REFRESH_STEPS:-10}" \
  --cache-encode-batch-size "${CACHE_ENCODE_BATCH_SIZE:-64}" \
  --cache-partition-graphs 1 \
  --query-target-sizes "${QUERY_TARGET_SIZES:-20,50,50,100,100}" \
  --query-size-jitter "${QUERY_SIZE_JITTER:-5}" \
  --prob-k-hop "${PROB_K_HOP:-0.35}" \
  --prob-single-part "${PROB_SINGLE_PART:-0.10}" \
  --prob-multi-coarse "${PROB_MULTI_COARSE:-0.25}" \
  --prob-random-walk "${PROB_RANDOM_WALK:-0.15}" \
  --prob-degree-k-hop "${PROB_DEGREE_K_HOP:-0.10}" \
  --validation-queries "${VALIDATION_QUERIES:-50}" \
  --validation-interval "${VALIDATION_INTERVAL:-5}" \
  --validation-seeds "${VALIDATION_SEEDS:-31415,27182}" \
  --validation-topks "${VALIDATION_TOPKS:-20,50,100,200,500,1000}" \
  "$@"

if [[ "${LIGHTNING_UPLOAD_RESULTS:-1}" == "1" ]]; then
  python - <<'PY'
import os
from pathlib import Path

from lightning_sdk.models import upload_model

cache_root = Path(os.environ.get("CACHE_ROOT", str(Path.cwd() / "cache"))).resolve()
name = os.environ.get(
    "LIGHTNING_RESULTS_MODEL",
    "swastik9895/financial-llm-training-project/jigsaw-rgcn-mag-results-latest",
)

if cache_root.exists():
    print(f"[LIGHTNING] Uploading cache/checkpoints/models from {cache_root} to {name}", flush=True)
    upload_model(name, path=cache_root, progress_bar=True)
    print("[LIGHTNING] Upload complete.", flush=True)
else:
    print(f"[LIGHTNING] Cache root not found, skipping upload: {cache_root}", flush=True)
PY
fi
