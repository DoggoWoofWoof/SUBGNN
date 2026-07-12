# Jigsaw

Jigsaw is a retrieval-constrained exact subgraph matching pipeline. A learned encoder retrieves candidate graph partitions, the overlap-cascade narrows the search region, and the Glasgow Subgraph Solver verifies exact matches inside the stitched candidate subgraph.

This repository is currently organized for the conference submission. Raw cloud run folders, model checkpoints, and cache artifacts are kept locally for provenance but ignored by Git; the reviewer-facing benchmark bundle is under `benchmarks/paper_results/final_results/`.

## Final Results

Canonical summary files:

- `benchmarks/paper_results/final_results/final_all_datasets_summary.csv`
- `benchmarks/paper_results/final_results/final_mag_summary.csv`
- `benchmarks/paper_results/final_results/final_arxiv_summary.csv`
- `benchmarks/paper_results/final_results/final_cora_summary.csv`
- `benchmarks/paper_results/final_results/final_benchmark_completion_audit.csv`

The final production matrix covers MAG, Arxiv, and Cora with two seeds, six production methods, eight query families, and query sizes 20/50/100. Each dataset summary has 288 rows, and `final_all_datasets_summary.csv` has 864 rows with full coverage reported in `production_grid_coverage.csv`. Arxiv also contains diagnostic ablations used to support method-design claims.

## Repository layout

```
paper/                         Manuscript: samplepaper.tex, figures (*.png), tables (table_*.tex), bib/cls.
benchmarks/paper_results/      Verified, reviewer-facing evidence backing the paper:
    final_results/             canonical summaries (final_all_datasets_summary.csv, per-dataset, audit).
    ablations/                 FullCov objective ablation + design ablation CSVs (back Tables 1 and the design table).
    diagnostics/               curated diagnostic CSVs referenced by the paper.
runs/diagnostics/              Generated diagnostic outputs + findings (candidate shrinkage, boundary overlap,
                               random_walk analysis, partition stats) and the source CSVs for several figures.
src/                           Canonical model (model.py: GIN + relation-aware RGCN) and data/eval.
scripts/                       Pipeline + tooling (see scripts/README.md):
    *.py (top level)           core pipeline, trainers, diagnostics, canonical figure/summary scripts.
    launchers/                 one-off Modal/Lightning run recipes (incl. staged walk-aware retrain).
    analysis/                  single-use analyses + manual Glasgow diagnostics.
    archive/                   superseded duplicates kept for provenance.
docs/                          Design docs, runbooks, and audits (see docs/README.md).
tests/                         pytest unit suite (pytest tests/).
```

## Key Scripts

- `scripts/benchmark_overlap_glasgow_cascade.py`: overlap-cascade benchmark (partition retrieval -> overlap -> prune -> Glasgow); includes selective/bridge overlap operators.
- `scripts/benchmark_glasgow.py`: Glasgow-based exact matching benchmark utilities.
- `scripts/lightning_production_benchmark.py`: Lightning launcher for final production benchmarks.
- `scripts/launchers/run_lightning_completion_jobs.ps1`: final Cora/Arxiv completion launcher.
- `scripts/launchers/run_overlap_model_benchmark_jobs.ps1`: Cora/Arxiv benchmark launcher for the overlap-trained GraphSAGE checkpoints.
- `scripts/launchers/run_lightning_mag_benchmark.sh`: Lightning runtime wrapper used by benchmark jobs.
- `scripts/summarize_production_benchmarks.py`: canonical per-query to summary aggregation.
- `scripts/generate_submission_figures.py`: canonical paper figure generation (from `benchmarks/paper_results/final_results/` + `runs/diagnostics/`).

## Reproducing Summaries

Raw per-query outputs are large and ignored under `runs/`. To regenerate a summary from a local result directory:

```powershell
$files = Get-ChildItem -Path runs\lightning_completion\<run>\results -Filter '*_per_query.csv' |
  Where-Object { $_.Name -notlike '*_partial_per_query.csv' } |
  ForEach-Object { $_.FullName }

python scripts\summarize_production_benchmarks.py @files --output runs\lightning_completion\<run>\summary.csv
```

The final published summaries are copied into `benchmarks/paper_results/final_results/` with SHA-256 hashes recorded in `manifest.json`.

For a paper-facing smoke check of the headline numbers and the newest selector/foreclosure claims, run:

```powershell
python scripts\analysis\reproduce_paper_numbers.py
```

The checker verifies the MAG deployed-encoder matrix, Cora/Arxiv label-selectivity selector, and MAG retrieval-remedy foreclosure from local CSVs. It treats non-core GNN-PE diagnostic provenance as optional unless `--strict-optional` is passed.

Budget reporting uses explicit columns: `first_solved_at_<B>` is the exact first-hit bucket, while `solved_by_<B>` is cumulative and should be used for budget-curve analysis.

## Exactness Scope

Jigsaw does not globally enumerate every subgraph isomorphism unless the retrieved stitched region covers the full target graph. The exactness claim applies to Glasgow verification inside the retrieved candidate region.

The primary retrieval metric is `FullCov@K`: every true partition touched by a query must appear within the retrieved top-K set. Recall alone is insufficient for downstream exact verification.
