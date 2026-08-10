# Benchmark Summary Index

The current paper-facing source of truth is the matched production bundle:

- `benchmarks/paper_results/final_results/HEADLINE_NUMBERS.csv`
- `benchmarks/paper_results/final_results/production_matched_costs.csv`
- `benchmarks/paper_results/final_results/production_matched_costs_by_family_size.csv`
- `benchmarks/paper_results/final_results/query_payload_v1_validation.json`

These files define the canonical 2,400-query workload per dataset: 1,800
planted positives and 600 audited negatives across two seeds, eight families,
and three sizes. Cora and Arxiv bounded policies are reported at half the
partition count; `FilterAll` is explicitly the exhaustive all-partition
ceiling. Candidate and cascade-cost columns use the same reporting endpoint as
the solve rate.

The following normalized grids are retained as legacy/full-grid diagnostics,
not as the source for paper headline rows:

- `benchmarks/paper_results/final_results/final_all_datasets_summary.csv`
- `benchmarks/paper_results/final_results/final_mag_summary.csv`
- `benchmarks/paper_results/final_results/final_arxiv_summary.csv`
- `benchmarks/paper_results/final_results/final_cora_summary.csv`
- `benchmarks/paper_results/final_results/final_benchmark_completion_audit.csv`
- `benchmarks/paper_results/final_results/csv_validation_report.json`

Ablation evidence is in:

- `benchmarks/paper_results/ablations/`

Regenerate paper tables from the matched production files and run
`validate_paper_budget_fairness.py`; do not copy headline values from the
legacy full-grid summaries. The validation reports confirm explicit
dataset/seed fields, no blank final rows, and no duplicate summary slices.

Budget reporting note: `first_solved_at_<B>` is an exact first-hit bucket; `solved_by_<B>` is cumulative and should be used for paper budget curves. The older `solved_at_<B>` columns are retained only for backward compatibility.

The overlap-aware GraphSAGE reruns remain diagnostic checkpoint studies rather
than the paper's matched production matrix. Their artifacts are:

- `runs/lightning_completion/jigsaw-cora-overlap-graphsage-bench-gcp-cpux8-v3`
- `runs/lightning_completion/jigsaw-arxiv-overlap-graphsage-bench-gcp-cpux8-v3`
- `runs/diagnostics/overlap_graphsage_benchmark_findings.md`
- `runs/diagnostics/overlap_graphsage_benchmark_aggregate.csv`
- `runs/diagnostics/overlap_graphsage_benchmark_budget_curve.csv`
- `runs/diagnostics/overlap_graphsage_benchmark_by_query_type_size.csv`

Both datasets validate at 2,400 logical queries per method/checkpoint label
across two seeds. Their full-coverage results are checkpoint diagnostics, not
the matched half-budget solve rates in the production table.

Query-family interpretation note: the current `multi_coarse` family is a
disconnected multi-region diagnostic, not the headline connected-query family.
Raw audits show that `FilterAll` has full node coverage and zero timeouts on
this family but still solves only about 2-3%, so failures there are attributed
to query construction / connected-component verifier semantics rather than
ordinary retrieval misses. The practical connected/local positive claim should
use `single`, `k_hop`, `degree_k_hop`, `random_walk`, and `multi_fine`.

Current expanded manuscript assets:

- `fig_jigsaw_pipeline.png`: retrieval-constrained exact-verification pipeline.
- `fig_encoder_architecture.png`: shared GIN/RGCN retrieval encoder architecture.
- `fig_fullcov_ablation.png`: matched Arxiv FullCov objective gain.
- `fig_production_positive_rates.png`: three-dataset production solve-rate overview, with and without the diagnostic `multi_coarse` family.
- `fig_mag_tradeoff.png`: MAG candidate-size/solve-rate tradeoff.
- `fig_jigsaw_query_family_heatmap.png`: Jigsaw query-family outcomes.
- `fig_jigsaw_budget_curves.png`: cumulative solved-by-budget curves.
- `fig_design_ablation.png`: matched half-partition MAG/Arxiv operator-necessity diagnostic.
