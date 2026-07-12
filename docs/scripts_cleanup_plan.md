# scripts/ Cleanup & Consolidation Plan

Read-only investigation. This document proposes a target layout; it changes no code.
Generated 2026-06-21.

## Ground-truth anchors (verified this pass)

- **Load-bearing imports (must stay importable):**
  - `benchmark_retrieval.py` <- imported by `benchmark_non_neural.py`, `benchmark_overlap_glasgow_cascade.py`
  - `benchmark_glasgow.py` <- imported by `benchmark_overlap_glasgow_cascade.py` (`import benchmark_glasgow as bench`)
  - `retrieval_strategies.py` <- imported by `benchmark_overlap_glasgow_cascade.py` **and `tests/test_retrieval_strategies.py`**
  - `benchmark_overlap_glasgow_cascade.py` <- imported by `probe_selective_overlap.py`
  - `coverage_losses.py` <- imported by **`tests/test_coverage_losses.py`**
  - `summarize_production_benchmarks.py` <- invoked by **`tests/test_summarize_production_benchmarks.py`**
  - `modal_train_graphsage.py` <- imported by `train_final_loss_local.py` (`from scripts.modal_train_graphsage import train`)
- **Git reality:** only `train_jigsaw_model.py` and `train_jigsaw.ipynb` are tracked (and modified). Every other script is untracked (working-tree only). Deleted/archived already: `scripts/train.py`, `scripts/train_jigsaw_arxiv.py`, `scripts/old/train_v1.py`, `scripts/old/train_v2.py`.
- **Dangling launchers:** `modal_retrieval_benchmark.py` and `modal_benchmark_glasgow.py` (the targets of 10 `run_*` launchers) **do not exist anywhere in the repo**. Those launchers cannot run as-is; they survive only as run recipes/documentation.

## Per-file table

| file | lines | one-line purpose | bucket | supersedes / superseded-by | action |
|---|---|---|---|---|---|
| `benchmark_glasgow.py` | 2289 | Main local Glasgow-solver benchmark generator (canonical paper CSVs) | KEEP-CORE | imported by overlap cascade | keep at top level |
| `benchmark_retrieval.py` | 2054 | Retrieval coverage benchmark + shared helpers (`FIXED_KS`, `coverage_metrics`) | KEEP-CORE | imported by non_neural, overlap cascade | keep at top level |
| `benchmark_overlap_glasgow_cascade.py` | 1951 | Overlap-cascade benchmark (partition+1-hop -> prune -> solver) | KEEP-CORE | imported by probe_selective_overlap | keep at top level |
| `retrieval_strategies.py` | 494 | Pure selection strategies (hybrid boundary expand, RRF) | KEEP-CORE | imported by cascade + tests | keep at top level |
| `coverage_losses.py` | 142 | Coverage-focused training losses (cvar/smoothmax/mean) | KEEP-CORE | imported by tests; logically belongs to trainer | keep at top level |
| `benchmark_non_neural.py` | 338 | Random/degree non-neural retrieval baselines (uses `benchmark_retrieval`) | KEEP-CORE | depends on benchmark_retrieval | keep at top level |
| `train_jigsaw_model.py` | 2845 | **Canonical trainer** (only git-tracked trainer; has Flickr support) | KEEP-TRAINER | canonical per both audits | keep at top level |
| `modal_train_graphsage.py` | 3434 | GraphSAGE/RGCN baseline trainer (`encoder_kind` flag); imported by local wrapper | KEEP-TRAINER | imported by train_final_loss_local; backs encoder-transfer claim | keep at top level |
| `train_graphsage_baseline.py` | 2832 | **99.5% identical to `train_jigsaw_model.py`** (15-line diff = only Flickr removed) | ARCHIVE-SUPERSEDED | superseded by `train_jigsaw_model.py` | confirm, then archive |
| `train_final_loss_local.py` | 170 | Thin local/Lightning wrapper calling `modal_train_graphsage.train.local(...)` | KEEP-TRAINER | depends on modal_train_graphsage | keep at top level |
| `lightning_rgcn_mag_train.py` | 123 | CLI helper to upload/launch Lightning MAG RGCN scratch training | KEEP-TRAINER (launcher-ish) | one of a kind | keep top level or scripts/launchers/ |
| `generate_submission_figures.py` | 508 | **Canonical** paper figures from `final_results/` + `runs/diagnostics/` | KEEP-FIGURE/SUMMARY | canonical per final_submission_audit | keep at top level |
| `generate_paper_plots.py` | 374 | Older figure set from legacy `paper_results/*_all.csv` | ARCHIVE-SUPERSEDED | superseded by `generate_submission_figures.py` (different/older data era) | confirm, then archive |
| `summarize_production_benchmarks.py` | 252 | **Canonical** summarizer over `*_per_query.csv` (test-covered) | KEEP-FIGURE/SUMMARY | canonical | keep at top level |
| `summarize_paper_benchmarks.py` | 117 | Summarizer over legacy `*_all.csv` -> markdown | ARCHIVE-SUPERSEDED | superseded by `summarize_production_benchmarks.py` | confirm, then archive |
| `rebuild_final_production_summaries.py` | 458 | **One-time** rebuild of the `final_results/` bundle + manifests | ARCHIVE-ONEOFF | distinct from canonical summarizer (not redundant) | confirm, then archive |
| `calc_metrics.py` | 44 | Ad-hoc table dump from two hardcoded CSVs to `results_grouped.txt` | ARCHIVE-ONEOFF | overlaps summarizers; hardcoded paths | confirm, then archive |
| `analyze_candidate_shrinkage.py` | 349 | Candidate-cascade shrinkage diagnostic (recently added) | KEEP-DIAGNOSTIC | per task ground truth | keep (scripts/ or scripts/analysis/) |
| `compute_boundary_overlap_stats.py` | 139 | Correct 1-hop boundary-overlap stats (recently added) | KEEP-DIAGNOSTIC | per task ground truth | keep (scripts/ or scripts/analysis/) |
| `probe_selective_overlap.py` | 242 | Offline recall/size probe of selective overlap (recently added) | KEEP-DIAGNOSTIC | imports cascade; per task ground truth | keep at top level (importer) |
| `analyze_benchmark_failures.py` | 147 | Summarize Glasgow benchmark CSV: coverage vs solve | ARCHIVE-ONEOFF | single-use analysis | confirm, then move to scripts/analysis/ |
| `analyze_final_ablation_retrieval.py` | 293 | Aggregate locked 2-seed matched-budget retrieval eval | ARCHIVE-ONEOFF | single-use analysis | confirm, then move to scripts/analysis/ |
| `analyze_multiview_retrieval.py` | 90 | Aggregate paired fixed-vs-multiview retrieval (McNemar) | ARCHIVE-ONEOFF | single-use analysis | confirm, then move to scripts/analysis/ |
| `compare_benchmark_csvs.py` | 167 | Run-level compare of two Glasgow benchmark CSVs | ARCHIVE-ONEOFF | reusable but ad-hoc | confirm, then move to scripts/analysis/ |
| `compare_training_progress.py` | 102 | Compare training logs by epoch loss/LR | ARCHIVE-ONEOFF | reusable but ad-hoc | confirm, then move to scripts/analysis/ |
| `test_glasgow_labels.py` | 198 | Diagnostic: does Glasgow enforce label matching (manual `__main__`, no asserts) | MOVE-TO-TESTS or ARCHIVE-ONEOFF | not pytest-shaped | confirm: likely scripts/analysis/ (not real test) |
| `test_glasgow_remote.py` | 57 | Smoke test of `src.glasgow_solver.glasgow_solve` (1 assert, `/app` path) | MOVE-TO-TESTS | belongs with tests/ | confirm, then move to tests/ |
| `lightning_cli_windows.py` | 33 | Windows shim for lightning-sdk CLI (`simple_term_menu` stub) | KEEP (util) | infra shim | keep at top level |
| `lightning_mag_benchmark.py` | 374 | Package+launch Lightning MAG benchmark job | KEEP / launcher | MAG-specific of the pair | keep top level or scripts/launchers/ |
| `lightning_production_benchmark.py` | 418 | Dataset-generic Lightning benchmark launcher (companion to MAG) | KEEP / launcher | generic superset of mag variant | keep top level or scripts/launchers/ |
| `run_mag_benchmark_matrix_local.py` | 257 | Non-Modal MAG cascade matrix runner (one job, internal workers) | KEEP / launcher | distinct local port | keep top level or scripts/launchers/ |
| `modal_eval_graphsage.py` | 813 | Modal eval of model on all 4 query types (FullCov@k) | KEEP / launcher | one of a kind | keep top level or scripts/launchers/ |
| `train_jigsaw.ipynb` | 201 | Interactive training notebook (git-tracked; contains creds per audit) | KEEP (notebook) | leave unchanged per audit | keep at top level |
| `run_cross_dataset_final_modal.ps1` | 166 | Launch cross-dataset final training (-> train_jigsaw_model.py) | ARCHIVE-ONEOFF | run recipe | move to scripts/launchers/ |
| `run_final_objective_ablation_modal.ps1` | 108 | Launch objective-ablation training (-> train_jigsaw_model.py) | ARCHIVE-ONEOFF | run recipe | move to scripts/launchers/ |
| `run_v7_training_modal.ps1` | 90 | Launch v7 training (-> train_jigsaw_model.py) | ARCHIVE-ONEOFF | run recipe | move to scripts/launchers/ |
| `run_v6_screen_modal.ps1` | 61 | Launch v6 screen training (-> train_jigsaw_model.py) | ARCHIVE-ONEOFF | run recipe | move to scripts/launchers/ |
| `run_graphsage_baseline_modal.ps1` | 80 | Launch GraphSAGE baseline (-> modal_train_graphsage.py) | ARCHIVE-ONEOFF | run recipe | move to scripts/launchers/ |
| `run_graphsage_final_loss_modal.ps1` | 185 | Launch GraphSAGE final-loss (-> modal_train_graphsage.py) | ARCHIVE-ONEOFF | run recipe | move to scripts/launchers/ |
| `resume_migrated_modal_runs.ps1` | 87 | Resume migrated Modal runs (-> train_jigsaw_model.py) | ARCHIVE-ONEOFF | run recipe | move to scripts/launchers/ |
| `fetch_modal_arxiv_v7_models.ps1` | 92 | Fetch v7 Arxiv models from Modal volume | ARCHIVE-ONEOFF | run recipe | move to scripts/launchers/ |
| `run_lightning_completion_jobs.ps1` | 124 | Launch Lightning completion jobs from package dir | ARCHIVE-ONEOFF | run recipe | move to scripts/launchers/ |
| `run_lightning_mag_benchmark.sh` | 367 | Bash launcher for Lightning MAG benchmark | ARCHIVE-ONEOFF | run recipe | move to scripts/launchers/ |
| `run_lightning_rgcn_mag.sh` | 66 | Bash launcher for Lightning RGCN MAG train | ARCHIVE-ONEOFF | run recipe | move to scripts/launchers/ |
| `run_production_benchmark_matrix.ps1` | 137 | Launch production matrix -> **missing `modal_benchmark_glasgow.py`** | ARCHIVE-ONEOFF (DANGLING) | dead target | move to scripts/launchers/ |
| `run_cross_dataset_retrieval_modal.ps1` | 102 | Launch retrieval -> **missing `modal_retrieval_benchmark.py`** | ARCHIVE-ONEOFF (DANGLING) | dead target | move to scripts/launchers/ |
| `run_final_ablation_retrieval_modal.ps1` | 63 | Launch retrieval -> **missing `modal_retrieval_benchmark.py`** | ARCHIVE-ONEOFF (DANGLING) | dead target | move to scripts/launchers/ |
| `run_graphsage_retrieval_modal.ps1` | 58 | Launch retrieval -> **missing `modal_retrieval_benchmark.py`** | ARCHIVE-ONEOFF (DANGLING) | dead target | move to scripts/launchers/ |
| `run_hybrid_retrieval_modal.ps1` | 38 | Launch retrieval -> **missing `modal_retrieval_benchmark.py`** | ARCHIVE-ONEOFF (DANGLING) | dead target | move to scripts/launchers/ |
| `run_multiview_retrieval_modal.ps1` | 65 | Launch retrieval -> **missing `modal_retrieval_benchmark.py`** | ARCHIVE-ONEOFF (DANGLING) | dead target | move to scripts/launchers/ |
| `run_retrieval_only_modal.ps1` | 32 | Launch retrieval -> **missing `modal_retrieval_benchmark.py`** | ARCHIVE-ONEOFF (DANGLING) | dead target | move to scripts/launchers/ |
| `run_v6_retrieval_modal.ps1` | 52 | Launch retrieval -> **missing `modal_retrieval_benchmark.py`** | ARCHIVE-ONEOFF (DANGLING) | dead target | move to scripts/launchers/ |
| `run_v7_retrieval_modal.ps1` | 56 | Launch retrieval -> **missing `modal_retrieval_benchmark.py`** | ARCHIVE-ONEOFF (DANGLING) | dead target | move to scripts/launchers/ |

## Proposed target structure

```
scripts/
  # KEEP-CORE pipeline (importable, do not move - relative imports)
  benchmark_glasgow.py
  benchmark_retrieval.py
  benchmark_overlap_glasgow_cascade.py
  benchmark_non_neural.py
  retrieval_strategies.py
  coverage_losses.py
  # KEEP-TRAINER
  train_jigsaw_model.py            (canonical)
  modal_train_graphsage.py         (baseline encoder, imported by wrapper)
  train_final_loss_local.py        (wrapper)
  lightning_rgcn_mag_train.py
  # KEEP-FIGURE/SUMMARY (canonical)
  generate_submission_figures.py
  summarize_production_benchmarks.py
  # KEEP-DIAGNOSTIC (recent)
  analyze_candidate_shrinkage.py
  compute_boundary_overlap_stats.py
  probe_selective_overlap.py       (imports cascade - keep adjacent)
  # KEEP infra/util/notebook
  lightning_cli_windows.py
  train_jigsaw.ipynb
  lightning_mag_benchmark.py
  lightning_production_benchmark.py
  run_mag_benchmark_matrix_local.py
  modal_eval_graphsage.py

  analysis/        # one-off / reusable-but-ad-hoc analyses
    analyze_benchmark_failures.py
    analyze_final_ablation_retrieval.py
    analyze_multiview_retrieval.py
    compare_benchmark_csvs.py
    compare_training_progress.py
    test_glasgow_labels.py         (diagnostic, not a pytest)

  launchers/       # one-off Modal/Lightning run recipes (.ps1/.sh)
    run_*.ps1, run_*.sh, resume_migrated_modal_runs.ps1,
    fetch_modal_arxiv_v7_models.ps1

  archive/         # superseded, kept for provenance
    train_graphsage_baseline.py    (superseded by train_jigsaw_model.py)
    generate_paper_plots.py        (superseded by generate_submission_figures.py)
    summarize_paper_benchmarks.py  (superseded by summarize_production_benchmarks.py)
    rebuild_final_production_summaries.py  (one-time bundle rebuild)
    calc_metrics.py                (ad-hoc, hardcoded paths)
```
Move `test_glasgow_remote.py` to repo-root `tests/`.

> **Import caveat:** the KEEP-CORE modules use *bare* relative imports
> (`import benchmark_glasgow`, `from benchmark_retrieval import ...`). Moving any
> of them into a subdir would break those imports unless run with the subdir on
> `sys.path`. Do **not** relocate any KEEP-CORE/importer file. Launchers and the
> trainer wrapper reference scripts by path string (`scripts/...`), so moving
> their *targets* would break the launchers too — that's why all KEEP-TRAINER and
> benchmark entrypoints stay at top level.

## Duplicate / superseded sets (canonical keeper named)

1. **Trainers:** keep `train_jigsaw_model.py`. `train_graphsage_baseline.py` is a
   15-line-diff copy (only Flickr support removed) — **archive**. `modal_train_graphsage.py`
   is genuinely different (`encoder_kind=graphsage/rgcn`, env-tunable threads) and
   is imported by `train_final_loss_local.py` — **keep**. The "GraphSAGE baseline"
   *paper claim* is served by `modal_train_graphsage.py` (the encoder flag), not by
   `train_graphsage_baseline.py`.
2. **Figures:** keep `generate_submission_figures.py` (reads canonical
   `final_results/`). `generate_paper_plots.py` reads the older `*_all.csv` era —
   **archive**.
3. **Summaries:** keep `summarize_production_benchmarks.py` (test-covered, per-query
   inputs). `summarize_paper_benchmarks.py` (legacy `*_all.csv`) — **archive**.
   `rebuild_final_production_summaries.py` is a *one-time* bundle/manifest rebuilder,
   not a redundant summarizer — **archive as one-off**, not a merge.
4. **Benchmark launchers:** `lightning_production_benchmark.py` (generic) is the
   superset of `lightning_mag_benchmark.py` (MAG-hardcoded); `run_mag_benchmark_matrix_local.py`
   is the distinct non-Modal port. No code merge recommended — different runtimes.
5. **NO merge** of `calc_metrics.py` into the summarizers (different output format,
   hardcoded paths) — archive instead.

## Safe to execute immediately (unambiguous)

- **Create directories** `scripts/launchers/`, `scripts/analysis/`, `scripts/archive/`.
- **Move all `run_*.ps1`, `run_*.sh`, `resume_migrated_modal_runs.ps1`,
  `fetch_modal_arxiv_v7_models.ps1`** into `scripts/launchers/` (they are run
  recipes; 10 already reference deleted targets and cannot run regardless). Update
  any doc that cites their old path.
- **Move `test_glasgow_remote.py`** into repo-root `tests/` (real smoke test).
- **Archive `generate_paper_plots.py`** (superseded by `generate_submission_figures.py`).
- **Archive `summarize_paper_benchmarks.py`** (superseded by `summarize_production_benchmarks.py`).
- **Archive `calc_metrics.py`** (ad-hoc, hardcoded paths).

## Confirm first (risky / uncertain)

- **`train_graphsage_baseline.py`** — 99.5% identical to canonical trainer but is
  *not* git-tracked. Confirm no launcher/run still targets it before archiving.
  (`run_graphsage_baseline_modal.ps1` targets `modal_train_graphsage.py`, not this
  file — supports archiving, but verify against any external/Modal job history.)
- **`rebuild_final_production_summaries.py`** — touches the *published* paper bundle
  (`final_results/manifest.json`). Archive only after confirming the bundle is final
  and won't be regenerated.
- **`test_glasgow_labels.py`** — named like a test but is a manual diagnostic (no
  asserts, `__main__`). Decide tests/ vs scripts/analysis/ (recommend analysis/).
- **Moving the analysis one-offs** (`analyze_*`, `compare_*`) into `scripts/analysis/` —
  none are imported by other modules (verified), but they may be referenced by path
  in docs/launchers; grep docs before moving.
- **Do NOT move** any KEEP-CORE module, `probe_selective_overlap.py`, the trainers,
  `generate_submission_figures.py`, or `summarize_production_benchmarks.py` — bare
  relative imports and path-string references will break.
- **`modal_eval_graphsage.py` / `lightning_*` / `run_mag_benchmark_matrix_local.py`** —
  classified KEEP-launcher; if you instead move them to `launchers/`, confirm nothing
  imports them (none do) and that their internal `scripts/` path references still resolve.
