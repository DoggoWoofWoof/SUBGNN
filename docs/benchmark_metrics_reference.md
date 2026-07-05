# Benchmark Metrics Reference

This document describes the active Glasgow benchmark schema produced by `scripts/benchmark_glasgow.py`.

Canonical current paper CSVs live in `benchmarks/paper_results/`. New Modal runs are written to the Modal volume under `results/` and should be copied into a named benchmark release folder with a manifest before being used in the paper.

## Query Fields

| Column | Meaning |
| --- | --- |
| `query_name` | Unique generated query label. |
| `query_type` | One of `single`, `multi_fine`, `multi_coarse`, `k_hop`. |
| `query_size` | Target query size requested for generation, usually 20, 50, or 100. |
| `query_nodes` | Actual generated query node count. |
| `true_coarse_indices` | Ground-truth coarse partitions touched by the query. |
| `anchor_coarse_idx` | Minimum/anchor true coarse partition, used for grouping/debugging. |
| `dataset` | Dataset key, e.g. `corafull`, `arxiv`, `pubmed`, `physics`, `citeseer`. |

## Retrieval Fields

For paper tables, prioritize `fullcov_at_k` / `retrieval_complete_at_k` over average recall. A single missed true partition can make exact verification impossible, so average `coarse_recall_at_k` is a secondary diagnostic rather than the main success criterion.

| Column | Meaning |
| --- | --- |
| `faiss_top_k` | Number of coarse partitions retrieved by FAISS for this run. |
| `coarse_seed_k` | Seed retrieval budget used for FAISS top-K. In dynamic runs this is the initial retrieval budget, e.g. `20`, not the final expanded candidate budget. |
| `coarse_score_k` | Optional deeper FAISS score pool used only for tie-breaking expansion, not as the headline seed retrieval budget. |
| `coarse_score_pool_fullcov` | Whether the optional score pool contains all true coarse partitions. |
| `coarse_score_pool_missed` | True coarse partitions missing from the score pool. |
| `predicted_coarse_idx` | Top-1 coarse partition prediction. |
| `correct_coarse_predicted` | Whether top-1 is in the true coarse partition set. |
| `fullcov_at_k` | Primary retrieval metric: whether every true coarse partition is present in top-K. Alias of `retrieval_complete_at_k` in new benchmark CSVs. |
| `retrieval_complete_at_k` | Backward-compatible name for FullCov@K. |
| `coarse_recall_at_k` | Fraction of true coarse partitions retrieved within the configured `faiss_top_k`. |
| `coarse_recall_at_20` | Recall@20 when `faiss_top_k >= 20`; otherwise `-1`. |
| `predicted_fine_idx` | Top fine prediction inside the top coarse partition, or `-1` when unavailable. |
| `true_fine_count` | Number of fine partitions touched by the query, when fine diagnostics are enabled. |
| `true_fine_ranks` | Rank of each true fine partition inside the fine candidate pool. Missing entries use `-1`. |
| `max_true_fine_rank` | Worst rank among found true fine partitions. Use with missed fields because missing true fine partitions are not counted in the max. |
| `fine_candidate_pool_count` | Number of fine partitions considered inside the retrieved coarse pool for fine stitching. |
| `fine_pool_fullcov` | Whether the full fine candidate pool contains every true fine partition. |
| `boundary_expand_coarse_budget` | Maximum number of coarse partitions allowed after graph-boundary expansion from the seed set. |
| `expanded_coarse_count` | Actual number of coarse partitions selected after boundary expansion. |
| `expanded_coarse_fullcov` | Whether the dynamically expanded coarse candidate set contains all true coarse partitions. |
| `expanded_missed_coarse` | True coarse partitions still missing after boundary expansion. |
| `mc_dropout_passes` | Number of stochastic query-embedding passes used for MC-dropout retrieval diagnostics. |
| `mc_dropout_top_k` | Per-pass top-K used for MC-dropout retrieval. |
| `mc_dropout_seed_fullcov` | Whether the MC-dropout union seed contains all true coarse partitions. |
| `mc_dropout_seed_missed` | True coarse partitions missing from the MC-dropout union seed. |

## Verification Fields

| Column | Meaning |
| --- | --- |
| `solver_mode` | `stitch` for Jigsaw retrieval plus Glasgow; `full` for full-graph Glasgow. |
| `perfect_solution_found` | Whether Glasgow found at least one exact match in the target graph it was given. |
| `first_solution_accuracy` | Accuracy of the first returned mapping against the planted query node IDs. |
| `best_accuracy` | Best mapping accuracy among returned solutions. |
| `solution_num_for_best_accuracy` | Number of solutions explored when the best accuracy was observed. |
| `total_solutions_in_timeout` | Number of solutions returned before timeout/limit. |
| `solver_level` | Retrieval expansion level that found the solution, e.g. `top-1`, `top-5`, `top-20`, `top-50`, or `none`. |
| `stitched_nodes` | Number of nodes in the stitched candidate graph. |
| `solver_timed_out` | Whether Glasgow timed out on the attempted target graph. |
| `stitch_strategy` | Candidate construction strategy, e.g. `ranked`, `neighbor_rerank`, `coarse_boundary_expand`, `fine_ranked`, `fine_boundary`, or `fine_boundary_expand`. |
| `candidate_fullcov` | Whether the actual candidate passed to Glasgow is complete. For coarse stitching this means all true coarse partitions are selected; for fine stitching this means all true fine partitions are selected. |
| `candidate_coarse_fullcov` | Whether the candidate's selected coarse partitions cover all true coarse partitions. |
| `candidate_fine_fullcov` | Whether the candidate's selected fine partitions cover all true fine partitions. |
| `candidate_missed_coarse` | True coarse partitions missing from the candidate. |
| `candidate_missed_fine` | True fine partitions missing from the candidate. |
| `pre_prune_stitched_nodes` | Candidate node count before optional label-based pruning. |
| `pruned_stitched_nodes` | Candidate node count after optional label-based pruning. |
| `prune_target_by_query_labels` | Whether candidate target nodes were filtered to labels present in the query before Glasgow. |
| `require_candidate_fullcov` | Whether solver execution was skipped until candidate FullCov was achieved. This is benchmark-only because real queries do not expose true partitions. |

## Production Summary Fields

The paper-facing CSVs under `benchmarks/paper_results/final_results/` are
grouped summaries, not raw per-budget rows. Each canonical production row is one
dataset/seed/model/method/query-type/query-size cell with 50 generated queries.

| Column | Meaning |
| --- | --- |
| `queries` | Number of generated logical queries in the group, normally 50. |
| `positive_queries` / `negative_queries` | Logical query polarity counts. Positive rows have planted matches; negative rows should return no match. |
| `positive_solved` | Positive logical queries with at least one exact solution found in the candidate graph. |
| `correct_no_match` | Negative logical queries with no solution and no timeout. |
| `false_positives` | Negative logical queries where a solution was returned. This is the main false-match count. |
| `timeouts` | Logical queries with at least one solver timeout across the budget sweep. |
| `avg_total_s` | Per-query end-to-end candidate construction plus solver time, averaged over logical queries. |
| `avg_candidate_nodes` | Candidate graph size after retrieval and pruning, averaged over logical queries. |
| `solved_at_<B>` | Backward-compatible exact first-hit bucket: number of logical queries first solved exactly at budget `B`. |
| `first_solved_at_<B>` | Explicit alias of `solved_at_<B>` for new analysis. |
| `solved_by_<B>` | Cumulative count of logical queries solved by budget `B`; use this for budget-curve tables and paper text. |

Canonical summaries skip rolling `*_partial_per_query.csv` files by default.
Partial files are checkpoints for Lightning resume and should not be mixed into
paper tables unless a recovery audit explicitly asks for them.

## Timing Fields

| Column | Meaning |
| --- | --- |
| `query_embedding_time` | Time to embed the query. |
| `faiss_coarse_search_time` | Time for coarse FAISS retrieval. |
| `faiss_fine_search_time` | Time for fine-level retrieval inside the top coarse partition. |
| `time_to_first_solution` | Glasgow time to first solution when found. |
| `time_to_best_solution` | Glasgow latency associated with the best returned mapping. |
| `solver_time` | Total Glasgow verification time for this row. |
| `total_time` | End-to-end row time including embedding, retrieval, stitching, and solving. |
| `model_load_time` | One-time model loading time copied onto each row. |
| `partition_time` | One-time hierarchy construction time copied onto each row. |
| `faiss_build_time` | One-time FAISS index construction time copied onto each row. |
| `solver_timeout` | Timeout used for each Glasgow call. |

## Interpreting Exactness

For `solver_mode=stitch`, Glasgow is exact inside the stitched candidate graph. The result is globally complete only if the stitched graph covers the true match region. For `solver_mode=full`, Glasgow is run on the whole graph and is the direct exact baseline where feasible.

K-hop queries should be interpreted as a bounded-retrieval stress test: they often span many coarse partitions, especially on Arxiv, so K sweeps are required to separate model retrieval quality from hard top-K capacity limits.

`multi_coarse` in the current paper release is a disconnected multi-region
diagnostic. It should not be used as the headline realistic connected
cross-boundary query family. Future generated `multi_coarse` caches must pass
the connected-query guard in `scripts/benchmark_overlap_glasgow_cascade.py`;
stale disconnected caches are invalid for new production claims.

For paper tables, label-pruned fine-boundary rows should be named explicitly.
They demonstrate that exact verification becomes fast once candidate coverage is
achieved, but they should not be described as pure retrieval without the pruning
qualification.

## Training Objective Notes

`coverage_topk` is a training-only knob for the partition coverage loss. It adds
a differentiable FullCov@K barrier on top of the all-positive coverage loss. For
a row with `P` true coarse partitions, the loss pushes every positive partition
above the negative threshold required for all `P` positives to fit inside the
effective top-K.

When `P` exceeds the requested `coverage_topk`, the effective K widens in
buckets of 10: top-20 for rows that fit in 20, top-30 for rows needing 21-30
partitions, top-40 for rows needing 31-40, and so on. This keeps broad k-hop
training examples useful without asking for impossible FullCov@20.

V6 adds `coverage_positive_aggregation`. `mean` preserves the older behavior;
`cvar` averages only the worst configured fraction of required positives; and
`smoothmax` is a differentiable approximation to the worst positive. Since one
missed required partition makes FullCov false, `cvar` or `smoothmax` is more
closely aligned with the paper's primary retrieval metric.

`max_live_positive_parts` controls how many true coarse partitions are
re-encoded with gradients in each batch. The remaining partition bank is still
cached. This hybrid keeps the full 200-partition comparison affordable while
allowing coverage gradients to update both query and positive-partition
representations.
