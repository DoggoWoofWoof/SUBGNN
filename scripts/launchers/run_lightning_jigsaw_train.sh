#!/usr/bin/env bash
# Generalized overlap-aware Jigsaw retraining launcher (Cora / Arxiv / MAG).
# Spot-safe: downloads the latest checkpoint from LIGHTNING_RESULTS_MODEL at start,
# resumes, trains with periodic checkpoint upload, and uploads the final model.
#
# Env knobs (with per-dataset defaults below):
#   DATASET                cora | arxiv | mag        (required)
#   ENCODER_KIND           graphsage | rgcn          (default graphsage; mag->rgcn)
#   EPOCHS, STEPS_PER_EPOCH, BATCH_SIZE, TRAINING_SEED
#   COVERAGE_TOPK, MAX_LIVE_POSITIVE_PARTS, MAX_TRAIN_COARSE_PARTS
#   LIGHTNING_RESULTS_MODEL  owner/teamspace/model-name (periodic + final upload, and resume source)
#   JIGSAW_RESUME          1 to download+resume from LIGHTNING_RESULTS_MODEL at start (default 1)
set -euo pipefail
# This script lives in scripts/launchers/, so the package root is two levels up.
cd "$(dirname "$0")/../.."

DATASET="${DATASET:?set DATASET=cora|arxiv|mag}"
export JIGSAW_TORCH_THREADS="${JIGSAW_TORCH_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$JIGSAW_TORCH_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$JIGSAW_TORCH_THREADS}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export JIGSAW_SKIP_MODAL_COMMIT="${JIGSAW_SKIP_MODAL_COMMIT:-1}"
export JIGSAW_UPLOAD_CHECKPOINTS="${JIGSAW_UPLOAD_CHECKPOINTS:-1}"   # periodic upload inside trainer

CACHE_ROOT="${CACHE_ROOT:-$PWD/cache}"
export CACHE_ROOT   # must be exported: the resume-download and final-upload python heredocs read os.environ["CACHE_ROOT"]
mkdir -p "$CACHE_ROOT"

# --- Per-dataset defaults (coarse parts: cora 20 / arxiv 200 / mag 2000) ---
case "$DATASET" in
  cora)  ENCODER_KIND="${ENCODER_KIND:-graphsage}"; COVERAGE_TOPK="${COVERAGE_TOPK:-10}";  MAX_LIVE_POSITIVE_PARTS="${MAX_LIVE_POSITIVE_PARTS:-20}"; MAX_TRAIN_COARSE_PARTS="${MAX_TRAIN_COARSE_PARTS:-20}" ;;
  arxiv) ENCODER_KIND="${ENCODER_KIND:-graphsage}"; COVERAGE_TOPK="${COVERAGE_TOPK:-50}";  MAX_LIVE_POSITIVE_PARTS="${MAX_LIVE_POSITIVE_PARTS:-64}"; MAX_TRAIN_COARSE_PARTS="${MAX_TRAIN_COARSE_PARTS:-80}" ;;
  mag)   ENCODER_KIND="${ENCODER_KIND:-rgcn}";      COVERAGE_TOPK="${COVERAGE_TOPK:-50}";  MAX_LIVE_POSITIVE_PARTS="${MAX_LIVE_POSITIVE_PARTS:-64}"; MAX_TRAIN_COARSE_PARTS="${MAX_TRAIN_COARSE_PARTS:-80}" ;;
  *) echo "[ERROR] unknown DATASET=$DATASET" >&2; exit 2 ;;
esac

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements_lightning_rgcn.txt

# --- Deterministic checkpoint path (must match train_final_loss_local.py) ---
RUN_NAME="${ENCODER_KIND}_final_loss_${DATASET}_seed${TRAINING_SEED:-7202}_overlap_topk${COVERAGE_TOPK}_live${MAX_LIVE_POSITIVE_PARTS}"
CKPT_PATH="$CACHE_ROOT/${DATASET}_${RUN_NAME}_checkpoint.pth"

# --- Spot-safe RESUME: pull latest checkpoint bundle from the results model ---
RESUME_ARG=()
if [[ "${JIGSAW_RESUME:-1}" == "1" && -n "${LIGHTNING_RESULTS_MODEL:-}" ]]; then
  echo "[RESUME] Attempting to download prior checkpoints from $LIGHTNING_RESULTS_MODEL into $CACHE_ROOT" >&2
  python - <<PY || echo "[RESUME] no prior model (first run) -- starting fresh" >&2
from lightning_sdk.models import download_model
import os
try:
    download_model(os.environ["LIGHTNING_RESULTS_MODEL"], os.environ["CACHE_ROOT"], progress_bar=False)
    print("[RESUME] download ok", flush=True)
except Exception as e:
    raise SystemExit(f"[RESUME] download failed: {e}")
PY
  # Pick the dataset's checkpoint if present (exact path, else newest matching)
  if [[ -f "$CKPT_PATH" ]]; then
    RESUME_ARG=(--resume-from-checkpoint "$CKPT_PATH")
    echo "[RESUME] will resume from $CKPT_PATH" >&2
  else
    FOUND=$(find "$CACHE_ROOT" -name "${DATASET}_*checkpoint*.pth" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    if [[ -n "$FOUND" ]]; then RESUME_ARG=(--resume-from-checkpoint "$FOUND"); echo "[RESUME] will resume from $FOUND" >&2; fi
  fi
fi

# --- Optional FINE-TUNE source: if no resume checkpoint was found above and a
#     RESUME_FROM_MODEL is given, start from its weights (fresh optimizer/schedule).
#     RESUME_MODEL_ONLY=1 (default) loads encoder weights only -- the fine-tune case.
#     RESUME_FROM_GLOB selects which .pth inside the model (default the best_fullcov one). ---
if [[ ${#RESUME_ARG[@]} -eq 0 && -n "${RESUME_FROM_MODEL:-}" ]]; then
  echo "[FINETUNE] Downloading resume-source model $RESUME_FROM_MODEL" >&2
  SRC_DIR="$CACHE_ROOT/resume_src"; mkdir -p "$SRC_DIR"
  LIGHTNING_RESUME_SRC="$RESUME_FROM_MODEL" RESUME_SRC_DIR="$SRC_DIR" python - <<'PY'
from lightning_sdk.models import download_model
import os
download_model(os.environ["LIGHTNING_RESUME_SRC"], os.environ["RESUME_SRC_DIR"], progress_bar=False)
print("[FINETUNE] download ok", flush=True)
PY
  SRC=$(find "$SRC_DIR" -name "${RESUME_FROM_GLOB:-*best_fullcov*.pth}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  [[ -z "$SRC" ]] && SRC=$(find "$SRC_DIR" -name "*.pth" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  if [[ -n "$SRC" ]]; then
    RESUME_ARG=(--resume-from-checkpoint "$SRC")
    [[ "${RESUME_MODEL_ONLY:-1}" == "1" ]] && RESUME_ARG+=(--resume-model-only)
    echo "[FINETUNE] starting from $SRC (model-only=${RESUME_MODEL_ONLY:-1})" >&2
  else
    echo "[FINETUNE] WARNING: no .pth found in $RESUME_FROM_MODEL; starting fresh" >&2
  fi
fi

python scripts/train_final_loss_local.py \
  --dataset "$DATASET" \
  --encoder-kind "$ENCODER_KIND" \
  --epochs "${EPOCHS:-90}" \
  --steps-per-epoch "${STEPS_PER_EPOCH:-100}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --training-seed "${TRAINING_SEED:-7202}" \
  --cache-root "$CACHE_ROOT" \
  --coverage-target-mode "${COVERAGE_TARGET_MODE:-overlap}" \
  --coverage-topk "$COVERAGE_TOPK" \
  --coverage-cvar-fraction "${COVERAGE_CVAR_FRACTION:-0.5}" \
  --max-live-positive-parts "$MAX_LIVE_POSITIVE_PARTS" \
  --max-train-coarse-parts "$MAX_TRAIN_COARSE_PARTS" \
  --checkpoint-interval-epochs "${CHECKPOINT_INTERVAL_EPOCHS:-2}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-5}" \
  --cache-refresh-steps "${CACHE_REFRESH_STEPS:-10}" \
  --cache-encode-batch-size "${CACHE_ENCODE_BATCH_SIZE:-64}" \
  --validation-queries "${VALIDATION_QUERIES:-50}" \
  --validation-interval "${VALIDATION_INTERVAL:-5}" \
  --validation-topks "${VALIDATION_TOPKS:-10,20,50,100,200}" \
  --prob-k-hop "${PROB_K_HOP:-0.35}" \
  --prob-single-part "${PROB_SINGLE_PART:-0.10}" \
  --prob-multi-coarse "${PROB_MULTI_COARSE:-0.25}" \
  --prob-random-walk "${PROB_RANDOM_WALK:-0.15}" \
  --prob-degree-k-hop "${PROB_DEGREE_K_HOP:-0.10}" \
  --query-target-sizes "${QUERY_TARGET_SIZES:-20,50,50,100,100}" \
  "${RESUME_ARG[@]}" \
  "$@"

# Final upload (periodic uploads already happened inside the trainer via JIGSAW_UPLOAD_CHECKPOINTS=1)
if [[ "${LIGHTNING_UPLOAD_RESULTS:-1}" == "1" && -n "${LIGHTNING_RESULTS_MODEL:-}" ]]; then
  python - <<'PY'
import os
from pathlib import Path
from lightning_sdk.models import upload_model
cache_root = Path(os.environ["CACHE_ROOT"]).resolve()
name = os.environ["LIGHTNING_RESULTS_MODEL"]
if cache_root.exists():
    print(f"[LIGHTNING] Final upload of {cache_root} -> {name}", flush=True)
    upload_model(name, path=cache_root, progress_bar=False)
    print("[LIGHTNING] Final upload complete.", flush=True)
PY
fi
