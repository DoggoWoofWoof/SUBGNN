# Multi-Vector Query Embeddings — Experiment Spec

## Status: Phase 0 completed, hypothesis falsified

The inference-only MaxSim probe has now been run on 1,800 MAG queries using the deployed encoder and
cached partition embeddings (`runs/multivector_probe/probe_multivector_per_query.csv`). It did
**not** clear the go/no-go gate. Overall FullCov@1000 moved from **50.1%** for the single-vector
baseline to **49.1%** for `subq8_max`, **49.7%** for `subq8_top2`, and **48.1%** for `subq4_max`.
The spatially-extended families stayed near-random:

| family | single | subq8_max | subq8_top2 | subq4_max |
|---|---:|---:|---:|---:|
| single | 98.3 [9] | 98.0 [11] | 98.3 [11] | 98.0 [12] |
| multi_fine | 99.7 [3] | 99.7 [6] | 99.7 [6] | 99.7 [6] |
| multi_coarse | 61.7 [821] | 56.7 [864] | 59.0 [870] | 51.3 [984] |
| degree_k_hop | 17.3 [1716] | 17.7 [1718] | 17.7 [1734] | 17.7 [1720] |
| k_hop | 14.7 [1716] | 14.3 [1722] | 14.7 [1726] | 14.3 [1716] |
| random_walk | 9.0 [1831] | 8.0 [1838] | 8.7 [1839] | 7.7 [1882] |

Entries are FullCov@1000 percentages with median worst-true-partition rank in brackets. The result
means the easy fix is not query-side aggregation: if the necessary information were already present
in the deployed embedding geometry, MaxSim over subqueries should have surfaced it. Future work
should be framed as learning embeddings that discriminate spatially-extended regions, not as merely
applying multi-vector inference to the current encoder. Do not proceed to Phase 1 training unless
there is a new representation-learning hypothesis strong enough to justify the spend.

**Original goal (now falsified by Phase 0).** Turn the paper's verified limitation into a learned-retrieval win: a single query vector
cannot sit near all 16–30 partitions a spatially-extended query spans, so the retriever is
near-random on k-hop / degree-k-hop / random-walk (median worst-true-partition rank ≈1700 of 2000,
FullCov@1000 ≈11–22%). A **multi-vector (set) query representation with MaxSim ranking** — a
partition scored by its best match to *any* query subvector — should rank true partitions much
higher on these families, closing the gap to contained families (single rank 7, multi-fine rank 3),
without regressing the easy families.

This is the ColBERT "late-interaction / MaxSim" idea applied to partition retrieval.

## Original Hypothesis (falsified)
Replacing `single query vector → cosine → partition` with `{query subvectors} → MaxSim → partition`
would raise FullCov@1000 on the three spatially-extended families by **≥20 absolute points** with **≤2 pt
regression** on single/multi-fine, and drops their median worst-true-partition rank from ≈1700 to
**<1000** — because coverage of a spread-out query is a *set* problem, not a *point* problem.

## Why it can be prototyped cheaply (key enabler)
`src/model.py::ImprovedSubgraphEncoder.forward` / `RelationAwareSubgraphEncoder.forward` already
return **`node_emb` (per-node, L2-normalized)** alongside the single `graph_emb`. So Phase 0 needs
**no retraining and no architecture change** — it reuses the deployed `seed7203` encoder and the
cached coarse-partition embeddings, and only changes the *ranking function*. And it needs **no
Glasgow, no candidate assembly, no apt build** — it is pure encode + similarity + argsort, i.e. a
~15-min job, not a multi-hour benchmark.

---

## Phase 0 — inference-only MaxSim (the go/no-go, ~free)

**What.** Add a retrieval-time ranker; measure ranking metrics only. No training, no solver.

**Query multi-vector construction (ablate three, cheapest first):**
1. **Node-set (ColBERT-style):** use the query's per-node `node_emb` directly as the set
   `{q_1..q_n}` (n = query size, ≤100). Zero extra encoding.
2. **Clustered:** k-means (cosine) the query node embeddings into `K∈{4,8,16}` centroids →
   `K` subvectors. Bounds the vector count.
3. **Subquery decomposition:** split the query into local subgraphs (BFS balls from a
   farthest-point sample of query nodes, radius chosen so pieces are contained), encode each with
   the *existing* encoder → one `graph_emb` per piece. Most faithful to "cover many regions."

**Ranking (ablate the aggregator):**
`score(partition p) = AGG_i sim(q_i, emb_p)` with `AGG ∈ {max (MaxSim), top-m mean (m=3), softmax-weighted}`.
Retrieve top-K partitions by `score`. Report also the ColBERT-style *per-true-partition best rank*
(for each true partition, its rank under its nearest query subvector).

**Metrics (the ones where it currently fails):** per-family median/mean `max_true_coarse_rank`,
FullCov@{20,100,1000}, and coarse-precision@budget (to catch false-partition inflation). Compare
head-to-head against single-vector neural and FeatureIndex on the *same* queries (both seeds, all
families/sizes, n=1800).

**Change-points (code):**
- `scripts/benchmark_overlap_glasgow_cascade.py`: add `rank_by_multivector(query, encoder, device,
  coarse_embeds, coarse_ids, mode, agg, K)` next to `build_faiss_ranking` (≈line 454). Reuse
  `encoder.forward(...)` `node_emb`; for subquery mode call the encoder per piece.
- A small `decompose_query(query, mode, K)` helper (BFS balls / k-means) — pure torch/networkx.
- A standalone `scripts/probe_multivector_ranking.py` that loads encoder + hierarchy + cached
  `*_coarse_embeddings.pt`, generates the query set, and dumps a per-query CSV with
  `max_true_coarse_rank` for {single-vector, node-set, clustered, subquery} — mirrors the columns in
  `runs/label_selectivity_experiments/` so the existing analysis scripts apply directly.

**Compute:** one short Lightning job (no `--solver`, no build) against
`jigsaw-mag-walkaware-training-package-v2` + a code-patch overlay, `--methods` = the probe, or run
`probe_multivector_ranking.py` directly. ~15 min, ~1 machine-hour. **This is the whole go/no-go.**

**Decision gate:** proceed to Phase 1 iff Phase 0 clears the hypothesis thresholds above. If MaxSim
inflates false partitions (precision collapses → candidate blow-up), fall back to top-m mean or the
clustered/subquery variants before abandoning.

---

## Phase 1 — train the retriever set-aware (only if Phase 0 wins)

**What.** Generalize the FullCov objective from "weakest single-vector positive" to
"weakest required partition under its best query subvector."

Current objective (`train_final_loss_local.py`, `coverage_positive_aggregation="cvar"`,
`--coverage-cvar-fraction`, `--coverage-topk`) optimizes `min/CVaR_{p∈P_i} s(z_Q, part_p)` with a
single query vector `z_Q`. Generalize to
`s_set(P_i∋p) = max_{k} s(q_k, part_p)` (MaxSim), then apply the same min/CVaR over required
partitions. Intuition: stop forcing one vector to be near everything; let each subvector own a
region and penalize the worst-covered region.

- Keep the multi-pool encoder; add the query-side subvector construction from Phase 0's winning
  mode. Optionally add a light learned "query-token" head that projects node/cluster embeddings into
  the retrieval space (a few hundred K params) — ablate learned-head vs reuse-`node_emb`.
- Train on MAG walk-aware distribution, same seeds; select checkpoint by set-FullCov on val.

**Compute:** one training run (comparable to the walk-aware retrain) + cache re-embed. GPU
(`L4`/`T4`) or `CPU_X_16`, a few hours.

## Phase 2 — full end-to-end (the paper win)

Run the full production matrix (solve rate, candidate, timeout, memory) with the trained
multi-vector retriever, **both label regimes**, vs single-vector neural + FeatureIndex + FilterAll.

**Success = the decisive end-to-end win the paper currently lacks:** the multi-vector learned
retriever beats FeatureIndex on **solve rate** (McNemar-significant) on ≥1 spatially-extended family
at coarse labels — ideally overall — while holding contained families. That converts
"honest characterized non-win" into "learned retrieval wins where a single-vector index/embedding
cannot," which is exactly what LoG/ECML reviewers said is missing.

---

## Baselines / ablations (one clean table)
single-vector neural (current) · FeatureIndex · MaxSim{node-set, clustered-K, subquery} ·
aggregator{max, top-m, softmax} · (Phase 1) trained-set-aware · (optional) multi-vector *partition*
embeddings.

## Risks & mitigations
- **False-partition inflation** (MaxSim ranks any partition near any subvector high → larger
  candidate, more solver load). Mitigate with top-m mean / softmax aggregation; **report candidate
  size + solve, not just FullCov.**
- **Vector-count cost** (node-set = up to 100 vectors × 2000 partitions). Cheap here, but clustered
  (K≤16) or subquery keeps it small and is the deployable form.
- **Partition side stays single-vector** — fine as a first cut (partitions are smaller/homogeneous);
  multi-vector partitions are a later ablation only if recall still caps.
- **Phase 0 could fail** (coverage limit is fundamental, not representational). Then the honest paper
  stands as-is and we've spent ~1 machine-hour to know it — cheap falsification.

## Actual payoff
Phase 0 did its job as a cheap falsification test. It ruled out inference-only multi-vector query
aggregation as the missing ingredient, prevented an expensive Phase 1 run on a weak premise, and
strengthened the paper framing: Jigsaw's limitation is embedding discriminability for
spatially-extended regions, while the submitted contribution remains retrieval-constrained exact
verification under bounded memory.
