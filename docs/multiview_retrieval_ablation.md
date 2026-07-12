# Multi-View Query Retrieval Ablation

## Question

Can the same trained Jigsaw model retrieve better partitions by encoding
connected parts of the query, combining how often each partition is retrieved,
and reranking partitions without retraining?

This experiment uses no query labels, true partition IDs, or Glasgow output
during retrieval.

## Implementation

For each connected 20-node k-hop query:

1. retain the complete query as the primary view;
2. choose deterministic farthest-first anchors inside the query;
3. construct up to six connected BFS views containing approximately 60% of
   the query nodes;
4. encode the full query and every view using the same checkpoint;
5. retrieve a complete coarse-partition ranking for every encoding;
6. fuse the rankings and select exactly the same K as fixed retrieval.

The implemented truth-free fusion variants are:

- `multiview_rrf`: reciprocal-rank fusion over the full query and all views;
- `multiview_occurrence`: prioritize partitions appearing in the top 20 of
  many views;
- `multiview_fusion`: balance full-query rank, occurrence support, mean view
  score, and maximum view score;
- `multiview_specialist`: strongly reward a partition ranked highly by one
  query view;
- `multiview_conservative`: mostly preserve the full-query ranking;
- `multiview_full_max`: combine only full-query evidence and the strongest
  individual-view evidence.

Relevant code:

- `scripts/retrieval_strategies.py`
- `scripts/benchmark_retrieval.py`
- `scripts/run_multiview_retrieval_modal.ps1`
- `scripts/analyze_multiview_retrieval.py`

## Locked Evaluation

The main comparison uses:

- the two clean final-objective checkpoints, seeds 7101 and 7102;
- locked query seeds 20260607 and 20260608;
- 100 queries per query seed;
- 400 paired model/query evaluations in total;
- identical candidate budgets for fixed and multi-view retrieval.

### FullCov Results

| Method | K=20 | K=50 | K=75 | K=100 |
| --- | ---: | ---: | ---: | ---: |
| Fixed full-query ranking | **104/400** | 195/400 | **270/400** | 328/400 |
| Multi-view fusion | 100/400 | 198/400 | 267/400 | **329/400** |
| Multi-view occurrence | 98/400 | **199/400** | 262/400 | 324/400 |
| Multi-view RRF | 95/400 | 198/400 | 267/400 | 324/400 |
| Multi-view specialist | 98/400 | 198/400 | 267/400 | 328/400 |

The best apparent multi-view improvements are small:

- occurrence at K=50: `199/400` versus fixed `195/400`;
- fusion at K=100: `329/400` versus fixed `328/400`.

### Paired Stability

| Variant | Budget | Wins over fixed | Losses to fixed | Net | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Occurrence | 50 | 10 | 6 | +4 | 0.4545 |
| Fusion | 50 | 6 | 3 | +3 | 0.5078 |
| Fusion | 100 | 6 | 5 | +1 | 1.0000 |

None of the gains are statistically clear. Multi-view reranking recovers some
queries, but it also breaks queries already solved by the complete-query
ranking.

## Refinement Checks

Two additional designs were evaluated on query seed 20260607 before deciding
whether they deserved confirmation on the second seed.

### Conservative 60% views

Giving 70-75% of the fusion weight to the complete-query ranking prevented
most damage, but it also removed almost all gains:

- `multiview_full_max`: equal to fixed at K=20, K=75, and K=100; `-1/200` at
  K=50;
- `multiview_conservative`: never improved FullCov at any tested budget.

### Smaller 35% views

Ten smaller connected views were intended to isolate local query regions.
They were generally worse:

- occurrence: `-11`, `-12`, `-8`, and `-9` FullCov at K=20/50/75/100;
- RRF: `-9`, `-11`, `-9`, and `-9`;
- the best isolated result was only `+2/200` at K=75 for `full_max` and
  `specialist`, with no statistical evidence.

The smaller fragments lose too much structural context and behave like noisy,
out-of-distribution queries for a model trained primarily on complete
connected query graphs.

## Comparison With Existing Retrieval

Using the same 400 clean-final evaluations:

| Retrieval method | Budget 50 | Budget 75 | Budget 100 |
| --- | ---: | ---: | ---: |
| Fixed full-query neural ranking | 195/400 | **270/400** | 328/400 |
| Locked boundary-dynamic retrieval | 187/400 | 264/400 | 321/400 |
| Best multi-view result at each budget | **199/400** | 267/400 | **329/400** |

The per-budget multi-view maxima are exploratory and choose different fusion
rules after observing results. They must not be presented as one locked
method. No single multi-view method consistently beats fixed retrieval.

## Decision

Keep **fixed full-query neural ranking** as the primary retriever.

Multi-view retrieval should remain an exploratory diagnostic because:

1. it provides no consistent FullCov improvement;
2. its small gains are not statistically clear;
3. it requires approximately one additional model encoding per query view;
4. view fragments can discard structural context and introduce noisy ranking
   changes.

The experiment does reveal a useful future direction: train the model with
explicit full-query/view consistency or learn a fusion calibrator using a
separate validation set. Without such training or calibration, averaging
occurrence across query parts is not better than the current fixed ranking.

## Reproduction

```powershell
.\scripts\run_multiview_retrieval_modal.ps1 `
  -Queries 100 `
  -Seeds @(20260607,20260608) `
  -ViewCount 6 `
  -ViewFraction 0.6 `
  -SupportDepth 20 `
  -Profile deepalimohapatra1973
```

```powershell
.\.venv_modal\Scripts\python.exe scripts\analyze_multiview_retrieval.py `
  runs\logs\retrieval_arxiv_khop_multiview_v6_f0p6_d20_q100_seed20260607_per_query.csv `
  runs\logs\retrieval_arxiv_khop_multiview_v6_f0p6_d20_q100_seed20260608_per_query.csv
```
