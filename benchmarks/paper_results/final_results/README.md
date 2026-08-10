# Final Benchmark Results

This directory is the paper-facing benchmark bundle for the conference submission.

> **HEADLINE NUMBERS — read this first.** The paper's headline solve rates
> (**Jigsaw: Cora 94.2 / Arxiv 92.8 / MAG 88.6**) come from the **deployed
> overlap-aware / walk-aware encoders** and are collected in
> **`HEADLINE_NUMBERS.csv`** (traced row-by-row in `../CANONICAL_SOURCES.md`).
>
> The per-config `final_*_summary.csv` files below are a legacy tidy grid whose
> production-policy rows are superseded for headline use. Their
> `model='mag_rgcn_best'` MAG neural rows are the earlier **seed7202** walk-aware
> checkpoint and aggregate to **86.0%**, *not* the deployed **seed7203** model's
> **88.6%**. Likewise the Arxiv `neural` rows are the pre-overlap-retrain base
> encoder (74.1% at the half budget, superseded by 92.8%). A canonical matched-
> budget rerun also replaces the earlier Cora/Arxiv cross-policy rows and supplies
> their missing candidate and timing columns. **Do not aggregate any
> `final_*_summary.csv` row for headline claims** -- use `HEADLINE_NUMBERS.csv`
> and `production_matched_costs.csv`.

Files:

- `HEADLINE_NUMBERS.csv`: **canonical paper-matching headline** (production matrix per dataset x method + MAG per-family), built from the deployed encoders. Reconciles exactly to the paper tables.
- `production_matched_costs.csv`: canonical Cora/Arxiv six-policy solve, terminal-candidate, cascade-time, and verifier-time rows at K=10/100 (FilterAll at K=20/200).
- `production_matched_costs_by_family_size.csv`: direct family-by-size evidence for solve-rate, candidate, and timing interpretations.
- `../ablations/full_budget_claim_audit.md`: classification of every paper-facing exhaustive-budget and 100% claim, including the corrected operator and scaling diagnostics.
- `../../../scripts/analysis/validate_paper_budget_fairness.py`: fail-fast paper audit for matched budget fractions, diagnostic denominators, populated matched-cost cells, and stale exhaustive-budget claims. Run it before rebuilding a submission PDF.
- `final_all_datasets_summary.csv`: combined legacy grid for MAG, Arxiv, and Cora (see caveat above for neural / Mean-RRF rows).
- `final_mag_summary.csv`: legacy MAG two-seed grid; neural / Mean-RRF rows are the superseded seed7202 walk-aware run (aggregate 86.0%).
- `final_arxiv_summary.csv`: legacy Arxiv two-seed grid; neural rows are the pre-overlap-retrain base `arxiv` encoder (superseded by 92.8%).
- `final_cora_summary.csv`: Cora two-seed grid.
- `final_diagnostic_model_ablation_summary.csv`: non-canonical diagnostic rows preserved for method evidence, including Arxiv design/v7 rows and MAG `mag_rgcn_final` model-variant rows.
- `final_benchmark_completion_audit.csv`: completion and schema audit for the final benchmark artifacts.
- `csv_cleaning_report.json`: record of blank, duplicate, and partial-row removals.
- `csv_validation_report.json`: invariant check for the canonical CSV bundle.
- `production_grid_coverage.csv`: expected-vs-actual production grid by dataset, method, model, seed, query type, target size, and polarity.
- `benchmark_grid_coverage_report.json`: coverage summary for final results and ablation CSVs.
- `query_payload_v1_validation.json`: serving-path migration audit over all 7,200 unique queries. It records exact pruning-token equivalence for all 5,400 positives, intentional payload-label divergence for all 900 label-corrupted negatives, and the validated Cora/Arxiv/MAG negative reruns.
- `manifest.json`: byte sizes and SHA-256 hashes for the CSVs above.

The final dataset summaries are production-only and use a shared schema with explicit `dataset` and `seed` columns. Each dataset has exactly 288 rows: 6 methods x 2 seeds x 8 query types x 3 query sizes, with 50 generated queries per row. The combined `final_all_datasets_summary.csv` has exactly 864 rows and is an exact row-wise union of the three dataset summaries. Canonical files contain no `_partial_per_query` rows; archived pre-clean/pre-normalization copies live under `archive/non_submission_20260620/` only for provenance.

The 216 negative summary rows were refreshed on 2026-08-01 using pruning tokens derived exclusively from each submitted query payload (`query_pruning_source=query_payload_v1`). The 648 positive summary rows were not rerun: the migration audit proves that their query-payload labels and signatures exactly equal the legacy evaluation tokens on all 5,400 unique positives. Superseded negative rows and their pre-merge manifests are preserved under `archive/query_pruning_pre_payload_v1_2026-08-01/`.

Budget columns are explicit: `first_solved_at_<B>` is the exact first-hit bucket for budget `B`, while `solved_by_<B>` is cumulative and should be used for budget curves. The older `solved_at_<B>` columns are retained as backward-compatible exact first-hit buckets.

Coverage note: the selected production grid is complete for Cora, Arxiv, and MAG. Extra diagnostic/model-variant rows are intentionally separated into `final_diagnostic_model_ablation_summary.csv` so they support ablation claims without inflating the canonical production matrix.

Raw Lightning/Modal run folders are intentionally kept out of this release bundle. They remain locally under `runs/` as provenance, while this folder contains the reviewer-facing summaries used for paper tables and claims.

For method-design evidence, use `../ablations/`. That folder contains the Arxiv design ablation extract, objective/loss ablation aggregates, paired tests, and v7 continuation summaries.
