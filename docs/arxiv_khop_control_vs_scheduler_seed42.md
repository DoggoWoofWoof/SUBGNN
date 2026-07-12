# Arxiv k-hop control vs scheduler comparison

Date: 2026-06-06

This note compares the finished Arxiv control model against the scheduler fine-tune branch on the same fixed k-hop probe:

- Dataset: OGBN-Arxiv
- Query type: `k_hop`
- Query generator: `aligned_connected_v2`
- Query count: 6
- Query size: 20
- Seed: 42
- Solver timeout flag: 45 seconds
- Modes: retrieved stitched graph plus oracle true-partition graph

## Artifacts

Training models:

- Control: `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v2_allpos_fresh.pth`
- Scheduler branch: `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v2_khop_sched_e60.pth`

Training logs:

- Control local: `runs/logs/coverage_v2_allpos_fresh_direct_20260606_011834.out.log`
- Scheduler local: `runs/logs/coverage_v2_khop_sched_e60_direct_20260606_050716.out.log`
- Control volume: `/cache/logs/train_arxiv_coverage_v2_allpos_fresh.log`
- Scheduler volume: `/cache/logs/train_arxiv_coverage_v2_khop_sched_e60.log`

Benchmark CSVs are stored in Modal volume under `/data/results/` and downloaded locally under `runs/logs/`.

## Final k-hop comparison

FullCov@K is the primary retrieval metric: a query is covered only if all true coarse partitions are present in the retrieved top-K set.

| Run | K | FullCov@K | Recall@K | Stitch solved | Oracle solved | Solver failures after FullCov | Avg nonzero stitched nodes | Median solver seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control `coverage_v2_allpos_fresh` | 20 | 1/6 (16.7%) | 0.6687 | 1/6 | 6/6 | 0 | 5,031.0 | 7.08 |
| control `coverage_v2_allpos_fresh` | 50 | 3/6 (50.0%) | 0.9046 | 3/6 | 6/6 | 0 | 30,414.7 | 42.11 |
| control `coverage_v2_allpos_fresh` | 100 | 4/6 (66.7%) | 0.9398 | 1/6 | 6/6 | 3 | 72,085.2 | 89.74 |
| scheduler `coverage_v2_khop_sched_e60` | 20 | 1/6 (16.7%) | 0.7533 | 1/6 | 6/6 | 0 | 5,058.0 | 7.69 |
| scheduler `coverage_v2_khop_sched_e60` | 50 | 2/6 (33.3%) | 0.8660 | 1/6 | 6/6 | 1 | 36,665.5 | 68.29 |
| scheduler `coverage_v2_khop_sched_e60` | 100 | 3/6 (50.0%) | 0.9243 | 1/6 | 6/6 | 2 | 71,997.2 | 79.58 |

## Training curve comparison

From epoch 62 onward:

| Run | Epochs parsed | Last epoch | Start loss | Last loss | Best loss | Tail average loss | Last LR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 59 | 120 | 7.734188 | 7.521412 | 7.152152 | 7.334795 | 5.0e-05 |
| scheduler | 58 | 120 | 10.767606 | 10.118533 | 9.866768 | 10.219505 | 5.0e-05 |

The scheduler branch used a different sample mix and higher partition emphasis, so its absolute loss is not directly comparable as a pure optimization score. The benchmark comparison is decisive: the scheduler branch did not improve the final k-hop FullCov or solved rate.

## Diagnosis

The oracle solver succeeds on every query for both models at every K. That means:

- Query construction is valid enough for these probes.
- Glasgow can solve the true-partition candidate region.
- The main bottleneck is retrieval and candidate construction, not the exact solver itself.

The control model is the better final Arxiv model:

- At K=50, control reaches FullCov@50 = 3/6 and stitch solved = 3/6.
- The scheduler branch reaches only FullCov@50 = 2/6 and stitch solved = 1/6.
- K=100 improves FullCov for both models but makes stitched graphs about 72k nodes, causing solver failures even when all true partitions are present.

Persistent control misses:

- At K=50, failed queries miss partitions `[116]`, `[21]`, and `[6, 97]`.
- At K=100, the remaining retrieval misses are `[116]` and `[97]`.
- This shows the model ranks most true partitions near the top, but one or two hard partitions still fall outside the retrieved set.

## Decision

Use `coverage_v2_allpos_fresh` as the current Arxiv model. Do not continue the `coverage_v2_khop_sched_e60` branch as the paper model.

## Follow-up: FullCov-gated deep retrieval

After the control-vs-scheduler comparison, we tested the control model with FAISS retrieval depth `K=200` and a benchmark-only guard that runs Glasgow only after the stitched candidate set contains all true coarse partitions.

Run tag:

- `coverage_v2_allpos_fresh_ranked_k200_fullcov_gated_seed42`

Result:

| Probe | Raw FullCov@200 | Candidate FullCov | Stitch solved | Oracle solved |
| --- | ---: | ---: | ---: | ---: |
| control ranked K=200, FullCov-gated | 6/6 | 6/6 | 2/6 | 6/6 |

Per-query first FullCov candidate levels:

| Query | Max true partition rank | First candidate FullCov level | Stitched nodes | Outcome |
| --- | ---: | --- | ---: | --- |
| `k_hop_0` | 122 | top-125 | 106,413 | timeout |
| `k_hop_1` | 83 | top-100 | 85,728 | timeout |
| `k_hop_2` | 42 | top-50 | 42,992 | timeout |
| `k_hop_3` | 35 | top-35 | 30,572 | solved |
| `k_hop_4` | 116 | top-125 | 106,537 | timeout |
| `k_hop_5` | 2 | top-20 | 17,587 | solved |

This confirms that the control model can retrieve every true coarse partition by `K=200` on this fixed probe. The remaining blocker is candidate size: whole coarse partition stitching creates 43k to 106k node targets for the hard queries, and Glasgow times out even when FullCov is achieved.

For labeled benchmarks, solver calls should be gated on candidate FullCov. If the candidate does not contain all true partitions, the solver cannot find the exact query and running it only wastes time. This is implemented via `--require-candidate-fullcov`.

For paper claims, report:

- FullCov@K as the headline retrieval metric.
- Recall@K as a secondary coverage-quality metric.
- Exact solver success on stitched candidates separately.
- Oracle true-partition success as a diagnostic upper bound.

For the June 8 submission, the safest claim is not "k-hop is solved." The defensible claim is:

1. The corrected all-positive coverage objective improves multi-partition retrieval.
2. On hard k-hop queries, increasing K improves retrieval coverage but creates a candidate-size tradeoff for exact verification.
3. Oracle rows show exact verification succeeds when the retrieved candidate region is correct and compact.

## Next engineering actions

Priority 1: add a compact candidate selection stage after top-100 retrieval.

- Retrieve top-100 coarse partitions.
- Re-rank or select a smaller subset, targeting FullCov while keeping stitched graph size closer to K=50.
- Candidate selection should be judged by FullCov, stitched nodes, and solver success.

Priority 2: add query-specific coverage calibration.

- The model should predict how many coarse partitions are needed, or stop adding partitions only when coverage confidence saturates.
- Blindly increasing K is not enough, because K=100 improves FullCov but harms verification.

Priority 3: run paper-scale fixed-seed tables after the compact selection patch.

- Use at least 30 to 50 queries per query family if time permits.
- Report per-family rows for `single`, `multi_fine`, `multi_coarse`, and `k_hop`.
- Include `K=20,50,100` plus oracle diagnostic for k-hop.

Priority 4: keep MAG code ready but do not move the main paper story to MAG until Arxiv is stable.

- MAG can be a future/generalization experiment.
- The current urgent blocker is Arxiv k-hop retrieval and candidate-size control.

## Follow-up: fine-boundary dynamic stitching

Whole-coarse stitching at high K recovers FullCov but creates 40k-100k node
targets. The next fix was to retrieve coarse partitions, then dynamically select
fine partitions within that retrieved coarse pool using the fine boundary graph.

Implemented benchmark settings:

- `--stitch-strategy fine_boundary`
- `--require-candidate-fullcov`
- `--prune-target-by-query-labels`
- oracle rows kept for diagnostic upper bound

Important interpretation: `--prune-target-by-query-labels` is a verifier-side
candidate reduction using the same discrete node labels that Glasgow receives.
It should be reported as label-pruned verification, not as pure neural retrieval.

### Seed-42, 6-query sanity probe

| Method | K | Candidate unit | Label prune | Stitch solved | Candidate FullCov | Oracle solved | Median solver seconds |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| coarse ranked | 50 | coarse | no | 3/6 | 3/6 | 6/6 | 42.11 |
| coarse ranked | 100 | coarse | no | 1/6 | 4/6 | 6/6 | 89.74 |
| fine ranked | 125 | fine | no | 1/6 | 1/6 | 6/6 | 0.02 |
| fine boundary | 125 | fine | no | 4/6 | 4/6 | 6/6 | 2.80 |
| fine boundary | 125 | fine | yes | 6/6 | 6/6 | 6/6 | 0.02 |

This shows that fine ranking alone is weak, but graph-boundary expansion over
the fine partitions is useful. Label pruning then makes the exact verifier fast
once the candidate is complete.

### Seed-42, 30-query k-hop probe

| Method | K | Query count | Stitch solved | FullCov@K | Candidate FullCov | Avg recall@K | Avg pre-prune nodes | Avg pruned nodes | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fine boundary + label prune | 125 | 30 | 26/30 (86.7%) | 26/30 | 26/30 | 0.9703 | 28,697.7 | 20.4 | 0 |
| fine boundary + label prune | 150 | 30 | 28/30 (93.3%) | 28/30 | 28/30 | 0.9860 | 36,784.2 | 20.6 | 0 |
| oracle true partitions | n/a | 30 | 30/30 (100%) | 30/30 | n/a | 1.0000 | n/a | n/a | 0 |

The remaining top-150 failures are retrieval misses before solver execution:

| Query | True coarse parts | Recall@150 | Missed coarse partitions |
| --- | ---: | ---: | --- |
| `k_hop_9` | 13 | 0.6923 | `[60, 127, 188, 193]` |
| `k_hop_20` | 9 | 0.8889 | `[176]` |

Conclusion: the immediate paper-safe method is not `K=200` all-partition
retrieval. It is dynamic fine-boundary candidate construction with FullCov-gated
diagnostics, plus label-pruned exact verification. The model still needs better
coarse/fine ranking for the two hardest k-hop cases, so a separate v3 fine
coverage fine-tune has been launched.

### v3 fine-coverage ablation

Completed run:

- run name: `coverage_v3_finecov_from_v2_e40`
- completed app id: `ap-wwDdgkeGtdFqSRqxETRT0B`
- completed function call: `fc-01KTDQGFFFC5T9S0NRCX9VH9D3`
- canceled first attempt: `ap-SRvasqlW9AITVIlFONNFf4` / `fc-01KTDPXQTNDD19JJDQAAHEWPVC`
- checkpoint: `/cache/arxiv_coverage_v3_finecov_from_v2_e40_checkpoint.pth`
- final model: `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v3_finecov_from_v2_e40.pth`
- volume log: `/cache/logs/train_arxiv_coverage_v3_finecov_from_v2_e40.log`
- local log: `runs/logs/train_arxiv_coverage_v3_finecov_from_v2_e40.log`

The run starts from the `coverage_v2_allpos_fresh` checkpoint, keeps
`gamma_partition=1.5`, and adds `gamma_fine_partition=0.5`. It finished
40/40 epochs without OOM or traceback. Final epoch summary: Avg Loss
`10.174313`, CoarsePart `4.299868`, FinePart `6.581912`, LR `5.0e-06`,
GPU memory `0.23/0.73 GB`.

Fixed-seed q30 comparison:

| Model | Method | K | Stitch solved | FullCov@K | Candidate FullCov | Oracle solved | Avg recall@K | Avg pre-prune nodes | Avg pruned nodes | Timeouts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 control | fine boundary + label prune | 125 | 26/30 | 26/30 | 26/30 | 30/30 | 0.9703 | 28,697.7 | 20.4 | 0 |
| v2 control | fine boundary + label prune | 150 | 28/30 | 28/30 | 28/30 | 30/30 | 0.9860 | 36,784.2 | 20.6 | 0 |
| v3 fine coverage | fine boundary + label prune | 125 | 24/30 | 24/30 | 24/30 | 30/30 | 0.9664 | 28,419.4 | 20.4 | 0 |
| v3 fine coverage | fine boundary + label prune | 150 | 26/30 | 26/30 | 26/30 | 30/30 | 0.9743 | 31,443.3 | 20.5 | 0 |

Conclusion: v3 is a useful ablation but should not replace the paper model.
Fine-coverage continuation reduced the candidate size at K=150, but it also
reduced FullCov and stitch success. Use `coverage_v2_allpos_fresh` for the
main Arxiv table unless a later run beats its fixed-seed FullCov@K.

## Follow-up: top-20 seed dynamic retrieval

The paper-safe retrieval story should start from a small neural seed set, not
from `K=125` or `K=150`. The corrected dynamic probe therefore fixes coarse
FAISS seed retrieval at `K=20`, then expands from those seeds through the coarse
partition boundary graph. Inside the expanded coarse pool it ranks fine
partitions and expands over the fine boundary graph.

Implementation note: `fine_boundary_expand` now means:

1. FAISS coarse seed retrieval with `--top-k 20`.
2. Coarse boundary expansion to a bounded coarse budget.
3. Fine candidate ranking inside that expanded coarse pool.
4. Fine boundary expansion before Glasgow verification.

This is different from the earlier `K=125/150` diagnostic. Here, `K=20` is the
retrieval seed; 50/75/100 are expansion budgets, not blind FAISS retrieval
depths.

Settings:

- model: `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v2_allpos_fresh.pth`
- queries: 30 fixed-seed `k_hop` queries, size 20, seed 42
- `--top-k 20`
- `--faiss-score-k 20`
- `--stitch-strategy fine_boundary_expand`
- `--stitch-seed-count 20`
- `--require-candidate-fullcov`
- oracle rows included

Corrected q30 comparison:

| Method | Coarse seed K | Coarse expansion budget | Label prune | FullCov@SeedK | Expanded FullCov | Candidate FullCov | Stitch solved | Oracle solved | Avg recall@SeedK | Avg pre-prune nodes | Avg pruned nodes | Timeouts |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top-20 seed + boundary expand | 20 | 50 | no | 9/30 | 14/30 | 14/30 | 14/30 | 30/30 | 0.6847 | 8,841.9 | 8,841.9 | 0 |
| top-20 seed + boundary expand | 20 | 75 | no | 9/30 | 17/30 | 17/30 | 16/30 | 30/30 | 0.6847 | 13,777.4 | 13,777.4 | 1 |
| top-20 seed + boundary expand | 20 | 100 | no | 9/30 | 20/30 | 20/30 | 16/30 | 30/30 | 0.6847 | 22,273.8 | 22,273.8 | 4 |
| top-20 seed + boundary expand | 20 | 50 | yes | 9/30 | 14/30 | 14/30 | 14/30 | 30/30 | 0.6847 | 8,841.9 | 20.3 | 0 |
| top-20 seed + boundary expand | 20 | 75 | yes | 9/30 | 17/30 | 17/30 | 17/30 | 30/30 | 0.6847 | 13,777.4 | 20.4 | 0 |
| top-20 seed + boundary expand | 20 | 100 | yes | 9/30 | 20/30 | 20/30 | 20/30 | 30/30 | 0.6847 | 22,273.8 | 20.6 | 0 |

Interpretation:

- The raw neural top-20 seed is too weak for hard Arxiv k-hop: FullCov@20 is
  only `9/30`.
- Dynamic boundary expansion is real and useful: at budget 100 it recovers
  `20/30`, an 11-query gain over the seed set.
- The remaining `10/30` failures are still retrieval misses after expansion,
  so the model/ranking objective remains the main bottleneck.
- Label pruning should not be treated as neural retrieval. It is a
  verifier-side diagnostic that reduces the candidate from thousands of nodes
  to about the query label support. Without it, the same b100 candidate has
  `20/30` FullCov but only `16/30` solved because Glasgow times out on 4 cases.
- MC-dropout seeding did not improve this fixed probe: 5 stochastic top-20
  passes still gave `9/30` MC seed FullCov and `20/30` final b100 FullCov.

Paper implication: do not headline the `K=150` result as the main method. Use
the top-20 seed dynamic expansion table to show the real retrieval/candidate
tradeoff, and keep label-pruned rows clearly marked as verifier-side
diagnostics. The model-side next step is to improve seed FullCov@20.

## Follow-up: v4 node-alignment continuation

Because the top-20 study shows seed ranking is the bottleneck, a new v4
continuation was started from the v2 checkpoint:

- run name: `coverage_v4_nodebeta_from_v2_e40`
- active app id: `ap-l8o8oBY4S2fJN7QKFzqbYV`
- source checkpoint: `/cache/arxiv_coverage_v2_allpos_fresh_checkpoint.pth`
- checkpoint: `/cache/arxiv_coverage_v4_nodebeta_from_v2_e40_checkpoint.pth`
- expected final model: `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v4_nodebeta_from_v2_e40.pth`
- expected volume log: `/cache/logs/train_arxiv_coverage_v4_nodebeta_from_v2_e40.log`

Key settings:

- `gamma_partition=1.5`
- `gamma_fine_partition=0.0`
- `alpha=0.15`
- `beta=0.10`
- `prob_k_hop=0.70`
- `prob_multi_coarse=0.22`
- `prob_single_part=0.03`
- `max_train_coarse_parts=100`
- `learning_rate=2e-5`
- plateau scheduler, minimum LR `5e-6`

The first epoch started cleanly with no OOM/traceback. Early sampling showed
coverage targets around avg/max `8.8/19` and `9.0/21`, which confirms that the
run is training on broad k-hop coverage rather than only easy one-partition
queries. Epoch 1 completed with Avg Loss `7.363656`, CoarsePart `4.403702`,
FinePart `0.000000`, LR `2.0e-05`, GPU memory `0.25/1.55 GB`.

After v4 finishes, benchmark it with the same top-20 seed dynamic sweep:

- `K=20`, `faiss_score_k=20`
- coarse expansion budgets 50, 75, 100
- pruned and unpruned variants
- oracle rows

Adopt v4 only if it improves FullCov@SeedK and expanded/candidate FullCov
against the v2 table above.

## Follow-up: v5 top-K barrier continuation

V5 is a separate model-side ablation started after v4. It keeps the same v2
checkpoint source but changes the partition loss so the optimization target
matches the benchmark metric more directly.

The older all-positive coverage loss says: every true coarse partition should
score higher than hard negatives. That helps recall, but it does not directly
ask whether all positives fit into the first `K` retrieved partitions. V5 adds a
FullCov@K barrier: for a row with `P` positive coarse partitions, each positive
is pushed above the negative threshold that would place all `P` positives inside
the effective top-K.

Effective K is bucketed:

- `P <= 20`: optimize for top-20
- `21 <= P <= 30`: optimize for top-30
- `31 <= P <= 40`: optimize for top-40
- and so on in buckets of 10

This matters because some broad k-hop training rows genuinely touch more than
20 partitions. Those rows should not be skipped, but they also should not be
forced into an impossible top-20 objective.

Run metadata:

- run name: `coverage_v5_topkbarrier_from_v2_e40`
- active app id: `ap-EIVIZ5xvoObQihlgdZyCAi`
- source checkpoint: `/cache/arxiv_coverage_v2_allpos_fresh_checkpoint.pth`
- checkpoint: `/cache/arxiv_coverage_v5_topkbarrier_from_v2_e40_checkpoint.pth`
- expected final model: `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v5_topkbarrier_from_v2_e40.pth`
- confirmed config: `coverage_topk=20`, `coverage_topk_weight=0.35`, `coverage_topk_margin=0.0`

Early health check: v5 resumed encoder weights from the completed v2 checkpoint,
reset optimizer/scheduler state, completed epochs 1-2 without OOM or traceback,
and is continuing training. It must be judged by the same fixed-seed top-20
dynamic sweep as v2 and v4: seed FullCov@20 first, then expanded/candidate
FullCov at budgets 50, 75, and 100.

## Final Retrieval-Only V2/V4/V5 Comparison

V4 and V5 completed successfully, and the final comparison was run without
Glasgow on the same 30 fixed-seed Arxiv k-hop queries.

| Model | Fixed FullCov@20 | Fixed FullCov@50 | Fixed FullCov@100 | Dynamic B50 | Dynamic B75 | Dynamic B100 | Avg Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 | 9/30 | 13/30 | **23/30** | **14/30** | 17/30 | 20/30 | 0.6847 |
| V4 | 9/30 | 15/30 | 20/30 | **14/30** | **18/30** | **21/30** | **0.7013** |
| V5 | 9/30 | **16/30** | 20/30 | 13/30 | 17/30 | **21/30** | 0.6879 |

Average expanded recall at dynamic budget 100:

- V2: `0.9248`
- V4: `0.9198`
- V5: `0.9198`

Fine-boundary FullCov at fine budget 100 after dynamic coarse budget 100:

- V2: `13/30`
- V4: `10/30`
- V5: `10/30`

Interpretation:

- The node-beta and top-K barrier continuations did not improve the metric that
  matters most for small neural retrieval: all three models remain at
  `9/30 FullCov@20`.
- V5 is best at fixed `K=50`, while V4 is best at dynamic budget 75. V4 and V5
  tie at dynamic budget 100 with `21/30`.
- The V4/V5 dynamic gain is not complementary: both recover the same one query
  beyond V2, so their dynamic-budget-100 union remains `21/30`.
- V2 remains the strongest broad fixed retrieval model (`23/30 FullCov@100`)
  and the strongest fine-level model. It should remain the main paper model.
- V4 and V5 should be reported as mixed or negative ablations, not replacements
  for V2.
- Dynamic expansion is validated as a retrieval technique, improving the best
  top-20 seed result from `9/30` to `21/30`, but the remaining `9/30` failures
  show that the model still does not provide complete coverage reliably.

Evidence:

- `runs/logs/retrieval_arxiv_khop_v2_v4_v5_q30_seed42_summary.csv`
- `runs/logs/retrieval_arxiv_khop_v2_v4_v5_q30_seed42_per_query.csv`
- `runs/logs/retrieval_arxiv_khop_v2_v4_v5_q30_seed42.remote.log`

## Hybrid neural-boundary and global-fine retrieval

A later retrieval-only sweep fixes the key limitation of the old dynamic
method: neural ranking was previously used only as a tie-breaker after the
top-20 seed. The new method combines normalized neural rank with boundary
support and periodically teleports to the strongest remaining neural
candidate. A second variant globally retrieves fine partitions, maps them to
coarse parents, and fuses that parent ranking with coarse retrieval.

Best fixed-seed q30 results:

| Method | FullCov@20 seed | FullCov B50 | FullCov B75 | FullCov B100 | Avg recall B100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old boundary expansion | 9/30 | 14/30 | 17/30 | 20/30 | 0.9248 |
| Coarse hybrid, weight 0.5, teleport 10 | 9/30 | 14/30 | 20/30 | **25/30** | 0.9601 |
| Global-fine RRF hybrid, weight 0.5, teleport 10 | **10/30** | 15/30 | **21/30** | **25/30** | **0.9626** |

This is the strongest current candidate-selection result and validates the
paper's smart-retrieval claim more directly than increasing blind K. The best
individual hybrid misses five queries; the union across tested hybrids covers
`26/30`, leaving room for a deterministic adaptive fusion policy.

## V6 CVaR/live-positive screening

V6 completed a 20-epoch model-only fine-tune from V2. It used CVaR over the
worst 25% of required coarse partitions, a dynamic FullCov top-K barrier, and
live re-encoding of up to 24 positive partitions per batch.

The run was healthy and reduced training loss from `6.9153` at epoch 1 to a
minimum of `6.2172` at epoch 10. However, fixed-validation FullCov@20 peaked at
only `9/30`, also at epoch 10. Later epochs occasionally improved average
recall, but did not improve FullCov@20.

The held-out seed-42 comparison then confirmed that V6-final is the strongest
current model:

| Model | Fixed FullCov@20 | Fixed FullCov@50 | Fixed FullCov@100 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| V2 | 9/30 | 13/30 | 23/30 | 0.9429 |
| V6 best-validation checkpoint | 10/30 | 15/30 | 25/30 | 0.9673 |
| **V6 final checkpoint** | **11/30** | **16/30** | **26/30** | **0.9761** |

The best single dynamic curve is V6-final coarse hybrid with model weight 0.5
and teleport interval 10: `11/30` at seed K=20, `17/30` at B50, `20/30` at
B75, and `26/30` at B100. Global-fine fusion can raise V6-final seed
FullCov@20 to `12/30`, but does not improve the best B100 result.

The 100-query, new-seed confirmation shows that the learned-ranking gain is
real but that V6 does not dominate every dynamic selector:

| Model/method | FullCov@20 | FullCov@50 | FullCov@75 | FullCov@100 |
| --- | ---: | ---: | ---: | ---: |
| V2 fixed | 24/100 | 48/100 | - | 81/100 |
| V6-final fixed | **25/100** | **53/100** | - | **85/100** |
| V2 best dynamic | 25/100 | 53/100 | 71/100 | **89/100** |
| V6-final best dynamic | **27/100** | **60/100** | **75/100** | 87/100 |

Use V6-final as the stronger learned ranker and constrained-budget model.
Retain V2 as an expansion ablation because its coarse hybrid remains stronger
at B100. Their failures are complementary: the union across deterministic V2
and V6 methods reaches `96/100` FullCov at B100. A deterministic cross-model
RRF evaluation is the final selector experiment.

## Final two-seed result

A deterministic cross-model selector fuses the complete V2 and V6 coarse
rankings using reciprocal-rank fusion, seeds from its top 20, then expands with
model weight `0.75` and neural teleport interval `10`. It never inspects true
partitions.

Aggregate over two independent 100-query seeds:

| Method | Seed/K20 | B50/K50 | B75 | B100/K100 |
| --- | ---: | ---: | ---: | ---: |
| V2 fixed neural ranking | 49/200 | 95/200 | - | 155/200 |
| **V6-final fixed neural ranking** | **53/200** | **106/200** | - | **163/200** |
| V2 coarse hybrid, weight 0.5, teleport 10 | 49/200 | 102/200 | 133/200 | 172/200 |
| **V2+V6 cross-model coarse RRF hybrid** | **53/200** | **109/200** | **141/200** | **176/200** |

The cross-model B100 result is `92/100` on seed 20260607 and `84/100` on seed
20260608. This is a modest repeatable gain over the same V2 hybrid comparator
(`89/100` and `83/100`), not evidence of near-perfect retrieval. Use this as
the strongest dynamic method, while reporting the added two-model inference
cost and retaining V6-final fixed retrieval as the cleanest model-quality
comparison.
