#!/usr/bin/env bash
# Walk-aware MAG RGCN retraining (staged; run when compute is available).
#
# WHY: random_walk is the weakest MAG family (neural solve 70.3%, coverage only
# 9.5% at budget 200; true partitions rank in the 500-2000 band). It is a
# RETRIEVAL-ranking problem, not an overlap problem (verified: one-hop overlap
# rescues 94% of k_hop misses but only 62% of random_walk misses; an overlap-side
# bridge-infill fix was falsified). The lever is training: the benchmarked model
# saw random walks only ~15% of the time and is biased toward compact queries, so
# it ranks diffuse, many-partition walk regions too low.
#
# WHAT THIS CHANGES vs the standard MAG recipe (run_lightning_rgcn_mag.sh):
#   - PROB_RANDOM_WALK 0.15 -> 0.35   (much more diffuse-walk training signal)
#   - PROB_DEGREE_K_HOP 0.10 -> 0.15  (more long-range structure)
#   - PROB_K_HOP 0.35 -> 0.20         (free up probability mass from compact balls)
#   - QUERY_TARGET_SIZES -> 50,100,100,100  (target the failing size-50/100 regime)
# Everything else (coverage loss, seeds, validation) is inherited unchanged so the
# result is comparable to the current production model.
#
# EVAL after training (cheap, targeted -- see docs/walk_aware_retrain_runbook.md):
#   benchmark only the random_walk family with the new model and compare the
#   coverage-by-budget table to the baseline (success = coverage@200 and solve up).
set -euo pipefail
cd "$(dirname "$0")/../.."

export PROB_RANDOM_WALK="${PROB_RANDOM_WALK:-0.35}"
export PROB_DEGREE_K_HOP="${PROB_DEGREE_K_HOP:-0.15}"
export PROB_K_HOP="${PROB_K_HOP:-0.20}"
export PROB_SINGLE_PART="${PROB_SINGLE_PART:-0.10}"
export PROB_MULTI_COARSE="${PROB_MULTI_COARSE:-0.20}"
export QUERY_TARGET_SIZES="${QUERY_TARGET_SIZES:-50,100,100,100}"
export QUERY_SIZE_JITTER="${QUERY_SIZE_JITTER:-8}"
export TRAINING_SEED="${TRAINING_SEED:-7203}"   # new seed so it doesn't overwrite the production model
export LIGHTNING_RESULTS_MODEL="${LIGHTNING_RESULTS_MODEL:-swastik9895/financial-llm-training-project/jigsaw-rgcn-mag-walkaware}"

echo "[walk-aware] PROB_RANDOM_WALK=$PROB_RANDOM_WALK QUERY_TARGET_SIZES=$QUERY_TARGET_SIZES seed=$TRAINING_SEED"
exec bash scripts/launchers/run_lightning_rgcn_mag.sh "$@"
