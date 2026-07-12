# Benchmark Summary Index

The current benchmark source of truth is the final conference bundle:

- `benchmarks/paper_results/final_results/final_all_datasets_summary.csv`
- `benchmarks/paper_results/final_results/final_mag_summary.csv`
- `benchmarks/paper_results/final_results/final_arxiv_summary.csv`
- `benchmarks/paper_results/final_results/final_cora_summary.csv`
- `benchmarks/paper_results/final_results/final_benchmark_completion_audit.csv`
- `benchmarks/paper_results/final_results/csv_validation_report.json`

Ablation evidence is in:

- `benchmarks/paper_results/ablations/`

This file intentionally does not duplicate table values. Regenerate paper tables directly from the normalized CSVs above to avoid stale copied numbers. The validation report confirms that the canonical CSVs have explicit `dataset`/`seed` fields, no blank or partial rows, no duplicate summary slices, and that `final_all_datasets_summary.csv` is exactly the union of the three dataset summaries.

Budget reporting note: `first_solved_at_<B>` is an exact first-hit bucket; `solved_by_<B>` is cumulative and should be used for paper budget curves. The older `solved_at_<B>` columns are retained only for backward compatibility.

Overlap-trained GraphSAGE postscript: the newer Cora/Arxiv overlap-aware GraphSAGE benchmark reruns are validated locally but are not yet folded into the canonical `benchmarks/paper_results/final_results/` bundle. Their artifacts are:

- `runs/lightning_completion/jigsaw-cora-overlap-graphsage-bench-gcp-cpux8-v3`
- `runs/lightning_completion/jigsaw-arxiv-overlap-graphsage-bench-gcp-cpux8-v3`
- `runs/diagnostics/overlap_graphsage_benchmark_findings.md`
- `runs/diagnostics/overlap_graphsage_benchmark_aggregate.csv`
- `runs/diagnostics/overlap_graphsage_benchmark_budget_curve.csv`
- `runs/diagnostics/overlap_graphsage_benchmark_by_query_type_size.csv`

Both datasets validate at 2,400 logical queries per method/checkpoint label across two seeds, with 1,800 positives solved, 600 negatives correctly rejected, and 100% positive full coverage for both `best_fullcov` and final checkpoints. Cora remains saturated and should be framed as a feasibility/cost check. Arxiv also reaches 100% end-to-end solves in this benchmark, so the paper should use these rows to discuss checkpoint choice and candidate/latency behavior rather than claiming a new solve-rate separation.

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
- `fig_design_ablation.png`: Arxiv pruning/component/exact-label diagnostic ablation.
