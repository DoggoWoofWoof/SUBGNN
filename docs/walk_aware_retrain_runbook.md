# Walk-Aware MAG Retraining — Runbook (staged; run when compute is available)

## Why
`random_walk` is the weakest MAG family: neural solve 70.3%, and true-match coverage is
only 9.5% at budget 200, rising to 47.6% at budget 1000 (it never plateaus). Diagnosis
(verified on the real `runs/lightning_mag_full` runs, matched hierarchy):
- It is **retrieval-ranking**, not overlap or solver: FilterAll solves random_walk at 96%,
  96.6% of failures are pre-solver, and one-hop overlap rescues only 62% of missed partitions
  (vs 94% for k_hop). An overlap-side "bridge-infill" fix was implemented and **falsified**.
- The encoder ranks the diffuse walk's true partitions in the 500–2000 band because training
  is biased toward compact, small (~20-node) queries and sees random walks only ~15% of the time.

## The fix
Retrain the MAG RGCN encoder with a walk-heavy, large-size query distribution so those
partitions rank into a small budget. Config deltas vs the production recipe are encoded in
`scripts/launchers/run_mag_walk_aware_retrain.sh`:

| knob | production | walk-aware |
|---|---|---|
| `PROB_RANDOM_WALK` | 0.15 | **0.35** |
| `PROB_DEGREE_K_HOP` | 0.10 | 0.15 |
| `PROB_K_HOP` | 0.35 | 0.20 |
| `QUERY_TARGET_SIZES` | 20,50,50,100,100 | **50,100,100,100** |
| `TRAINING_SEED` | 7202 | 7203 (new — keeps the production model intact) |

Everything else (coverage loss, validation, cache settings) is inherited from
`run_lightning_rgcn_mag.sh` so the result is directly comparable.

## Launch (when compute is available)
```bash
# on the Lightning/GCP box, from repo root:
bash scripts/launchers/run_mag_walk_aware_retrain.sh
# overridable via env, e.g. PROB_RANDOM_WALK=0.4 QUERY_TARGET_SIZES=100,100 bash scripts/launchers/run_mag_walk_aware_retrain.sh
```
Validation (FullCov@k on random_walk) prints every 5 epochs; watch random_walk FullCov@200.

## Evaluate cheaply (do NOT re-run the full grid)
Benchmark ONLY the random_walk family with the new checkpoint and compare to baseline:
```bash
python scripts/benchmark_overlap_glasgow_cascade.py --dataset mag --method hybrid \
  --query-types random_walk --target-sizes 20,50,100 --budgets 20,50,100,200,500,1000 \
  --prune-query-labels --component-solve --signature type_rel_feat32 --solver-timeout 5 \
  --model mag_rgcn_walkaware=<new_ckpt> --cache-dir <mag_cache> \
  --output-prefix runs/probes/mag_rw_walkaware
```
Then re-run the coverage-by-budget check (the snippet used in this analysis) on the output.

## Success criteria
- random_walk coverage@budget-200 rises materially above the current **9.5%** (target ≥ 30%).
- random_walk solve rate rises above the current **70.3%** without inflating candidate size
  (i.e. the true partitions now rank into a small budget, not via brute-force budget growth).
- No regression on the strong families (single/k_hop/degree_k_hop stay ≥ ~93%).

If coverage@200 does not move, the bottleneck is encoder capacity/features, not the query
mix — next step would be structural features or harder walk-tail negatives.
