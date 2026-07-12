# scripts/ layout

Cleaned 2026-06-21 (32 → 20 top-level scripts; one-offs and launchers organized into
subdirs; superseded duplicates archived). See `docs/scripts_cleanup_plan.md` for the full
rationale and the per-file table.

## Core pipeline (importable — do NOT move; they use bare relative imports)
- `benchmark_glasgow.py` — local Glasgow-solver benchmark + query generation.
- `benchmark_retrieval.py` — retrieval coverage benchmark + shared helpers (overlap index,
  signature/label tokens, mean features).
- `benchmark_overlap_glasgow_cascade.py` — **the main benchmark**: partition retrieval →
  one-hop (or selective) overlap → signature/label pruning → component solve → Glasgow.
- `benchmark_non_neural.py` — random/degree non-neural retrieval baselines.
- `retrieval_strategies.py` — selection strategies (hybrid boundary expand, RRF).
- `coverage_losses.py` — coverage-focused training losses (used by trainer + tests).

## Trainers
- `train_jigsaw_model.py` — canonical trainer (arxiv/cora; Modal app).
- `modal_train_graphsage.py` — RGCN/GraphSAGE encoder trainer (`encoder_kind`); the MAG path.
- `train_final_loss_local.py` — local/Lightning wrapper around `modal_train_graphsage.train`.
- `lightning_rgcn_mag_train.py` — Lightning MAG RGCN launch helper.

## Diagnostics (recent — overlap accounting / shrinkage / selective overlap)
- `analyze_candidate_shrinkage.py` — overlap→signature→label→component cascade analysis.
- `compute_boundary_overlap_stats.py` — correct per-partition boundary-overlap stats.
- `probe_selective_overlap.py` — offline ($0) probe of overlap policies (size/build-time;
  recall only valid if queries match the index's hierarchy — it warns otherwise).

## Figures / summaries (canonical, used by the paper)
- `generate_submission_figures.py` — paper figures from `final_results/` + `runs/diagnostics/`.
- `summarize_production_benchmarks.py` — summarizer over `*_per_query.csv` (test-covered).

## Benchmark launchers / eval / infra
- `lightning_production_benchmark.py`, `lightning_mag_benchmark.py`,
  `run_mag_benchmark_matrix_local.py`, `modal_eval_graphsage.py`, `lightning_cli_windows.py`,
  `train_jigsaw.ipynb`.

## Subdirectories
- `launchers/` — one-off Modal/Lightning run recipes (`run_*.ps1/.sh`, `fetch_*`, `resume_*`),
  including the staged **`run_mag_walk_aware_retrain.sh`** (see `docs/walk_aware_retrain_runbook.md`).
  Note: 10 legacy `*_retrieval_modal.ps1` / `run_production_benchmark_matrix.ps1` target
  `modal_retrieval_benchmark.py` / `modal_benchmark_glasgow.py` which no longer exist — kept as
  historical recipes only.
- `analysis/` — single-use analyses (`analyze_*`, `compare_*`) and manual Glasgow diagnostics
  (`test_glasgow_labels.py`, `test_glasgow_remote.py` — not pytest unit tests). The paper-facing
  checker is `analysis/reproduce_paper_numbers.py`; run it from the repo root to verify the
  deployed MAG matrix, cross-dataset selector, and retrieval-remedy foreclosure claims.
- `archive/` — superseded duplicates kept for provenance: `train_graphsage_baseline.py`
  (≈copy of `train_jigsaw_model.py`), `generate_paper_plots.py` (→ `generate_submission_figures.py`),
  `summarize_paper_benchmarks.py` (→ `summarize_production_benchmarks.py`),
  `rebuild_final_production_summaries.py` (one-time bundle rebuild), `calc_metrics.py` (ad-hoc).

Unit tests live in repo-root `tests/` (`pytest tests/` — 12 pass locally; Glasgow integration
needs the solver binary and lives in `scripts/analysis/`).
