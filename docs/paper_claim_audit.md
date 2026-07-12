# Paper Claim → Code/Data Verification Map

Audit of `paper/samplepaper.tex` against the repo (2026-06-21). Status: every load-bearing
claim is now traceable to a file. One substantive error was found and fixed (MAG framing).

## Corrected this pass
- **MAG is heterogeneous / relation-aware, NOT "homogenized with weak signal."** The paper
  said "homogenized MAG" / "weaker attributed signal after homogenization" in 3 places
  (abstract, encoder section, datasets). VERIFIED WRONG: `src/data.py:40-98` flattens OGBN-MAG
  to one index but **retains** `node_type`, `edge_type` (relation ids), node-type one-hot +
  per-relation-degree features (`feature_schema="mag_type_rel_v1"`), and `src/model.py:69-115`
  (`RelationAwareSubgraphEncoder`) is 6× `RGCNConv(num_relations, num_bases)` consuming
  `edge_type`. The 1,939,743 nodes are the full hetero OGBN-MAG (4 node types). Fixed the
  paper to describe MAG as heterogeneous, modeled relation-aware with RGCN.

## Verified claims (claim → evidence)
| Claim | Paper | Evidence |
|---|---|---|
| Encoder: 6 blocks, 256 hidden, 128-dim L2 output, multi-pool mean/max/sum + MLP + skip | Sec 4.1 | `src/model.py:7-65`; instantiated `ImprovedSubgraphEncoder(base, 256, 128)` `scripts/train_jigsaw_model.py:1711` |
| MAG uses relation-aware RGCN | Sec 4.1 | `src/model.py:69-115`; `scripts/modal_train_graphsage.py:2193` |
| Partition counts (Cora 20/100, Arxiv 200/1000, MAG 2000/10000) + median sizes | Tab partition_sizes | `runs/diagnostics/partition_size_stats.csv` |
| Boundary overlap: Cora 1,533/1.6×, Arxiv 2,470/2.9×, MAG 5,228/5.4×; neighbors 948; loose bound 921,494 | Tab partition_overlap, Sec overlap-scale | `runs/diagnostics/boundary_overlap_stats.csv`, `partition_overlap_stats.csv` |
| FullCov ablation: control fixed 21.7/39.0/66.0, worst-rank 74.02; final 26.0/48.8/82.0, worst-rank 58.97 | Tab loss_ablation | `benchmarks/paper_results/ablations/fair_ablation_locked_aggregate.md` (0.217/0.390/0.660/74.02; 0.260/0.4875/0.820/58.97) |
| McNemar p=9.8e-6 @50, 4.0e-10 @100 | Tab loss_ablation | `.../fair_ablation_locked_aggregate_paired.csv` (fixed: 9.775e-6, 3.988e-10) |
| Production matrix MAG (CORRECTED/merged): neural 86.0, FilterAll 98.4, Mean-RRF 84.5, MeanFeat 64.6, random 51.4, topo 37.1 | Tab production_matrix | recomputed from merged `final_all_datasets_summary.csv` (1548/1771/1521/1162/926/668 of 1800) — exact match |
| Cora/Arxiv 100% all methods | Tab production_matrix, family | same summary CSV |
| Candidate cascade: neural overlap 59%/pruned 84.5K; FilterAll 544K; overlap_fullcov==pruned_fullcov; build 3.4s vs solver 0.25s | Sec shrinkage | `runs/diagnostics/candidate_shrinkage_summary_magfull.csv` |
| random_walk: 70.3% solve, 94% vs 62% rescue, 96.6% pre-solver, coverage 2%@20→47.6%@1000 | Sec family | `runs/diagnostics/random_walk_analysis.md`, recomputed from `runs/lightning_mag_full/*neural*` |
| Design ablation (no-signature 46.9K candidate nodes, etc.) | Tab design_ablation | `benchmarks/paper_results/ablations/arxiv_design_ablation_summary.csv` (source located; spot-check values when finalizing) |

## Thorough recompute pass (every table/inline number)
All recomputed directly from CSVs (`/tmp/verify.py`, `/tmp/v2.py`):
- **Production matrix MAG** — exact match on every cell: neural 64.6%/cand 224,384/10.56s/54.2ms/neg 99.8%/FP 1/TO 20; FilterAll 72.4%/463,458/5.80s/264.1ms/TO 27; Mean-RRF 63.8%/218,686; MeanFeat 51.1%; random 40.7%; topo 27.2%.
- **MAG neural family** — exact: single 99.3, k_hop 93.3, degree_k_hop 91.7, random_walk 60.0, multi_fine 40.7, multi_coarse 2.3.
- **MAG neural budget curve** — exact: 28.3/36.4/42.0/48.5/57.3/64.6 by 20/50/100/200/500/1000.
- **MAG neural size slices** — exact: 66.8/65.3/61.5 at 20/50/100.
- **Cora/Arxiv candidate-node cells** — match (Cora neural 2,524≈2.5K, mean_rrf 2,458, mean_feat 3,549, random 9,516, topo 9,589, filterall 843; Arxiv all ≤287). Solver-ms cells differ slightly from the paper (Cora neural 5.8 vs 7.9 ms) — the paper's Cora/Arxiv rows use the corrected-rerun timing, not the stale canonical rows; reconciled by the merge below.

## ARTIFACT INCONSISTENCY — RESOLVED (2026-06-22)
The corrected connected multi_fine/multi_coarse rows for all three datasets are now MERGED into
`benchmarks/paper_results/final_results/final_all_datasets_summary.csv` (backup:
`final_all_datasets_summary.csv.pre_connected_merge`). 216 rows substituted, 864 total, 0
duplicate keys. The CSV now reproduces the paper directly:
- Cora/Arxiv = 100% for all methods (no longer 71.7%/73.5%).
- MAG corrected: multi_fine 40.7%->100%, multi_coarse 2.3%->71.7% (neural), lifting the MAG
  aggregate to neural 86.0% / FilterAll 98.4% (was 64.6% / 72.4%). Paper tables, family table,
  budget curves (35.6/.../86.0), size slices (92.3/86.2/79.5), and prose all updated to match;
  figures regenerated. No "provisional" caveats remain.
Merge inputs: `runs/lcr_mag_v3/summary.csv` (drop `mag_rgcn_final` diagnostic rows),
`runs/lcr_arxiv_v3/summary.csv`, `runs/lcr_cora_v3_summary.csv`. Script: `/tmp/merge.py`.

## Statistical significance (added 2026-06-22)
`scripts/analyze_significance.py` -> `runs/diagnostics/mag_significance.{csv,md}`, surfaced in
`tab:significance`. MAG solve rates reproduce the canonical grid exactly (n=1800/method) with
bootstrap 95% CIs; paired exact-McNemar shows Jigsaw significantly beats every baseline
including Mean-RRF (70/43, p=0.014) and is significantly below FilterAll (2/225, p=2.4e-64).

## Recommended residual checks before submission
- `tab:design_ablation`: source `arxiv_design_ablation_summary.csv` verified to exist and the
  key claim holds directionally (no-signature inflates candidate nodes to ~30-46K vs ~0.2K
  full). Exact per-cell aggregate is a diagnostic slice over 162 rows — document the
  aggregation subset (sizes/seeds/types) if a reviewer presses, or relabel as a single slice.
- `ref_gnnpe` is cited as "Neural Subgraph Isomorphism Counting" — confirm author/venue.
