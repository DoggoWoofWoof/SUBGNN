# Production Benchmark Protocol

This protocol avoids oracle gating. A query expands only because the previous
candidate returned no solution or timed out.

## Query Families

- single-partition connected fragments
- multi-fine fragments within one coarse partition
- multi-coarse fragments across neighboring coarse partitions
- random-anchor K-hop connected queries
- random-walk connected queries
- degree-stratified random-anchor K-hop connected queries
- label-corrupt negative/no-match queries
- structure-perturbed negative queries

Current K-hop queries are random-anchor local connected BFS blobs inside a
3-hop neighborhood. They are not uniform random induced subgraphs.

## Budgets

Report strict and fallback budgets.

| Dataset | Coarse parts | Strict | Expansion | 50 percent | Full/filter-all |
| --- | ---: | --- | --- | ---: | ---: |
| Cora | 20 | 2, 5, 10, 20 | 20 | 10 | 20 |
| Arxiv | 200 | 20, 50, 100 | 200 | 100 | 200 |
| MAG | 2000 | 20, 50, 100 | 200, 500 | 1000 | 2000 |

## Methods

- full graph solver, where feasible
- filter-all: all partitions -> signature/label prune -> components -> solver
- neural fixed retrieval
- neural hybrid cascade
- random cascade
- mean-feature cascade
- neural + mean-feature RRF cascade
- topology-feature cascade
- no-overlap ablation
- no-signature-filter ablation
- no-exact-label-filter ablation
- no-component-solve ablation
- component-solve final system

## Metrics

- solved / total
- solved at K=20/50/100/200/500/1000/2000
- timeout count
- unknown-within-budget count
- FullCov@K and recall@K for diagnostics
- Hit@K and Precision@K for diagnostics
- max true-partition rank and p50/p95/p99 rank
- candidate node containment and edge containment when available
- avg/p95 candidate nodes and candidate edges
- avg component-solver nodes
- component count and largest component size
- node and edge reduction factor versus full graph
- retrieval latency
- candidate construction/pruning latency
- solver latency
- total cascade latency with p50/p95/p99
- cold index construction time
- signature/exact-label cache construction time
- model/index/cache size
- CPU/GPU peak memory when available
- false-positive rate on negative queries
- correct no-match count after full fallback
- negative timeout / unknown-within-budget count

## Current Launch Shape

The production matrix uses two locked seeds, `20260607` and `20260608`, with
50 queries per query family per target size. Target sizes are `20,50,100`.
Positive query families are `single`, `multi_fine`, `multi_coarse`, `k_hop`,
`random_walk`, and `degree_k_hop`. Negative query families are
`negative_label` and `negative_structure`.

`negative_label` keeps the positive query structure but changes one node
feature to a target-absent label, so a labeled match should not exist.
`negative_structure` adds one non-edge to a positive query while preserving
labels; it is a structure-perturbation stress test and must be interpreted with
the full-fallback result, because another copy of that perturbed pattern could
exist elsewhere in the target graph.

## Claim Discipline

The exact solver is sound inside the candidate graph. Global completeness is
guaranteed only when the fallback includes all partitions or the full graph and
the solver finishes. Otherwise the production output is "unknown within budget",
not "no match".
