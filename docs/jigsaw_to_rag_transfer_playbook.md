# Retrieval Lessons from Jigsaw → RAG: A Transfer Playbook

> **How to use this doc.** Paste it into the new project's chat as context. It abstracts the
> hard-won lessons from *Jigsaw* (a retrieval-constrained exact subgraph-matching system) into
> **general retrieval principles**, then translates each into **concrete RAG implementation +
> experiments**. Every principle is backed by a real result (numbers in §7). Read §1 for the
> headline transfers, §3 for the how-to, §4 for the method, §6 for a starting plan.

---

## 1. TL;DR — the transferable wins

1. **No single retriever wins; route by an *offline selectivity* statistic.** Lexical/exact
   retrieval beats dense embeddings when query terms are *rare/near-unique*; dense wins when terms
   are *common/semantic*. Don't default to "hybrid fuse" — **measure selectivity offline and
   route**. (This is the BM25-vs-dense debate, settled by evidence.)
2. **Optimize *coverage of all required evidence*, not average relevance.** If an answer needs
   N passages and you miss ANY one, the answer is wrong. Train and evaluate on
   **"did we retrieve every required chunk"** (FullCov@k) and optimize the **weakest** required
   chunk (min/CVaR), not mean similarity.
3. **Inference-time tricks don't fix a representation gap — falsify them cheaply first.** Query
   rewriting, HyDE, decomposition, multi-vector/ColBERT, RAG-fusion **do not repair** a retriever
   that fundamentally can't localize diffuse evidence. We tested six such tricks; none moved the
   needle. Run a **1-hour go/no-go probe** before any expensive fix.
4. **Classify queries by evidence structure and report per-family.** "Answer in one passage" is
   easy; "answer spread thinly across many passages" (multi-hop / aggregation) is hard for
   single-vector retrieval — and the difficulty is *predictable* from evidence dispersion.
   Aggregate metrics hide this.
5. **Pipeline = retrieve → expand → lossless-prune → verify.** Coverage is the failure mode, not
   generation: if retrieval misses the evidence, the LLM answers wrong no matter how good it is.
6. **Treat the context budget as a recall/cost frontier**, not a fixed k. Expose the knob.
7. **Method that made all of the above trustworthy:** cheap falsification gates, adversarial
   verification of every finding, honest negative results as contributions, and a reproducibility
   bundle where every headline number maps to one script + one CSV.

---

## 2. The source project in one paragraph

Jigsaw does **exact subgraph matching at million-node scale under a memory budget**. A direct
exact solve on a 1.9M-node graph is infeasible (0/15 queries solved, whole-graph residence). Jigsaw
instead **retrieves** a small set of graph partitions with a GNN, **stitches** their boundary
overlap, **prunes** losslessly by cheap typed signatures + exact labels, and only then runs the
exact solver on the bounded candidate. The retriever is a *component*; the contribution is the
bounded-memory exact-verification *system*. Structurally this is **RAG for graphs**: retrieve the
relevant pieces, assemble a bounded context, then run an exact "reader" (the solver) instead of an
LLM. That's why the retrieval lessons port directly.

---

## 3. The wins, and how to port them to RAG

### 3.1 Coverage-conditioned retrieval — retrieve ALL required evidence, optimize the weakest

**What we found.** Standard retrieval training maximizes *average* positive similarity, which is the
wrong objective when an answer requires *several* items. We introduced **FullCov**: a query is
"covered" only if **every** required partition is in the top-K, and we optimized the
**worst-covered** required item (a min / CVaR-over-positives loss), not the mean. On a locked test
this lifted fixed-budget coverage from **66% → 82%** (FullCov@100) with a controlled ablation
(same seeds, same budget), and it was decisive exactly on the multi-item query families.

**General principle.** For multi-evidence answers, the metric that matters is
**all-required-recall** (FullCov@k = 1 iff every gold passage is retrieved), and the training
signal should penalize the **weakest** required passage, because that's the one that sinks the
answer.

**RAG translation — implement + test.**
- **Eval:** add **FullCov@k** (fraction of queries where *all* gold passages are in top-k)
  alongside recall@k / MRR. For most multi-hop/aggregation benchmarks this is the number that
  predicts end-answer correctness. Report it per query-family (§3.4).
- **Train:** if you fine-tune the retriever/embedder, change the loss from "mean over positives"
  to **min / CVaR over the query's gold passages** (contrastive on the *hardest* positive). Cheap
  change, directly targets the failure mode.
- **Rerank:** a cross-encoder reranker that's tuned for average nDCG can still drop a required
  passage; select the rerank cutoff by **FullCov**, not nDCG.

### 3.2 The retriever portfolio — route lexical-vs-dense by *offline* selectivity  ← biggest transfer

**What we found (the crossover).** Which retriever localizes best **flips entirely with label
selectivity**, and we confirmed it on three datasets:
- Under **near-unique** labels, a classical exact-label index is near-oracle and **beats** the
  learned dense retriever on every hard family (median rank of the worst required item: Cora **2 vs
  5**, Arxiv **4 vs 44**; MAG 96.7% vs 88.6% solve).
- Under **coarse/common** labels, the classical index **collapses toward random** (Arxiv **104 of
  200**) while the learned retriever degrades gracefully and **wins every family** (Arxiv **44 vs
  104**).
- The flip tracks a single **offline** statistic — *median partitions per label* (Cora 1 vs 12,
  Arxiv 1 vs 124.5). Per-query sign test: **p < 10⁻³⁰** in both regimes.

**General principle.** Sparse/lexical/exact retrieval dominates when the query's discriminative
tokens are **rare** (high selectivity → near-oracle); dense/semantic retrieval dominates when they
are **common/paraphrastic** (low selectivity). There is **no universal winner** — and blindly
hybrid-fusing both leaves performance on the table. The right move is a **portfolio selector**
routed by a statistic you can compute **without knowing the answer**.

**RAG translation — implement + test.** *This is the BM25-vs-dense question, but decided by data.*
- **Selector, not always-fuse.** Compute an **offline selectivity signal** per query — e.g., the
  IDF / corpus rarity of its content terms, or the number of documents its key entities appear in.
  Route **rare-term queries → BM25/lexical (or exact-filter)**, **common/semantic queries → dense**.
  Fuse only where they genuinely tie.
- **The offline signal is deployable and non-oracle** — it's a property of the query and the corpus
  index, not of the gold answer. (In Jigsaw it was partitions-per-label; in RAG it's
  doc-frequency / IDF / entity cardinality.)
- **Measure the crossover on a model-independent ranking metric first** (rank of the worst gold
  passage), then corroborate with end-answer accuracy — the ranking result is cleaner and doesn't
  confound the reader.
- **Deliverable framing:** "we don't claim dense retrieval wins everywhere; we select the best
  bounded retriever for the query's selectivity regime." That's more defensible *and* usually
  higher-performing than a fixed choice.

### 3.3 Foreclosure — inference-time tricks don't fix a representation gap (falsify cheaply)

**What we found.** The learned retriever was near-random on **spatially-extended** queries (evidence
spread thinly across ~16–30 partitions). We tested **six inference-time remedies** to fix it —
multi-vector/subquery MaxSim (ColBERT-style), finer-granularity retrieval, one- and two-hop
neighborhood expansion, dual-granularity late interaction, and score diffusion. **All failed**: the
best gain was **+2.3 points** against a **+20** bar; one made it worse. The mechanism: the index
graph was **high-degree at every granularity** (median neighbors 948/2000 coarse, 568/10000 fine),
so "expand to neighbors" floods rather than localizes. **Conclusion: the bottleneck is embedding
discriminability — fixable only by *training* a better representation, not by any query-side trick.**

**General principle.** When a retriever genuinely can't *represent* the distinction a query needs,
**inference-time query gymnastics won't save you.** They rearrange a signal that isn't there. The
honest fix is representation learning (better training data / objective / model) — and that's a
project, not a patch.

**RAG translation — implement + test.**
- Before investing in **query rewriting, HyDE, query decomposition, multi-vector/ColBERT, or
  RAG-fusion**, run a **cheap falsification probe** (inference-only, ~1 hour): apply the trick,
  measure FullCov@k on the hard family, and set a **go/no-go gate** (e.g., "+X points or we stop").
  Most of these tricks help *contained* queries and do **nothing** for genuinely diffuse
  (multi-hop/aggregation) ones — know which before you build.
- If the probe fails, the lever is **retriever fine-tuning** (better negatives, coverage loss,
  domain adaptation) or a **structurally different index** (e.g., graph/entity-linked retrieval for
  multi-hop) — not more prompt-side query expansion.
- **A tested, explained "this doesn't work" is a real result** — it closes the obvious reviewer/PM
  question "why didn't you just use HyDE/ColBERT?" and stops wasted effort.

### 3.4 The difficulty taxonomy — classify queries by evidence structure, report per-family

**What we found.** Query difficulty was **monotonically predicted** by *evidence dispersion*:
concentrated queries (≈50 nodes in one partition) hit **98%** coverage; diffuse queries (≈1.7 nodes
across ~29 partitions) fell to **9%**. Aggregate numbers hid this entirely; the per-family split was
what revealed the real failure mode and the density mechanism.

**General principle.** "Retrieval accuracy" is meaningless as a single number. Split by
**evidence structure**: (a) single-passage, (b) multi-passage within one document/region,
(c) multi-hop / cross-document / aggregation (evidence spread thin). Difficulty ≈ *(required
passages) × (1 / signal-per-passage)*.

**RAG translation.** Bucket your eval set into those three families and report FullCov@k / answer
accuracy **per bucket**. Expect single-passage to saturate and multi-hop/aggregation to lag — and
target effort there. This also tells you *which* fix class applies (rerank vs graph-index vs
retrain).

### 3.5 The pipeline pattern — retrieve → expand → lossless-prune → verify

**What we found.** Jigsaw = retrieve partitions → **expand** by boundary overlap → **prune**
losslessly by cheap signatures (never drops a true item) → **verify** exactly on the bounded
candidate. Soundness lives in the verifier; **the only failure mode is retrieval coverage.**

**General principle / RAG translation.**
- **Expand:** after top-k, pull in **structurally adjacent** context — neighboring chunks,
  parent document, or graph/entity-linked passages (this is "sentence-window" / "parent-document" /
  GraphRAG). Expansion recovers *boundary* misses cheaply (in Jigsaw it fixed 94% of near-miss
  k-hop cases — but only 62% of truly diffuse ones; expansion has limits, see §3.3).
- **Lossless prune:** apply cheap **filters that provably never drop a required item** (metadata
  filters, exact must-match terms) *before* the expensive reranker/LLM — shrinks the candidate
  without hurting coverage.
- **Verify:** the LLM/reader only sees the bounded candidate. **If retrieval missed the evidence,
  the answer is wrong regardless of the model** — so measure and attribute failures to *coverage*
  vs *generation* separately. (Caveat vs Jigsaw: the LLM reader is **not sound** — it can
  hallucinate even with correct context — so add answer-grounding/citation checks; see §5.)

### 3.6 Budget as a recall/cost/context frontier

**What we found.** The retrieval budget K is not a fixed setting but a **tunable frontier**: small K
= cheap, low recall; large K = higher recall, more cost; the exhaustive setting is a ceiling, not a
deployable point. Jigsaw also never held the whole graph resident (2.4 GB streaming vs 10.2 GB, a
4.2× reduction).

**RAG translation.** Treat **top-k / context-window budget** as an explicit recall–cost–latency
frontier and expose it: a caller needing high recall pays more context; one needing speed accepts a
quantified miss rate. "Stuff the whole corpus in context" is the ceiling, not a strategy — bound
what you load, and report the frontier (FullCov@k vs tokens/latency).

---

## 4. The methodology playbook (how to run the RAG project rigorously)

These are *process* wins that made the results above trustworthy — port them wholesale.

- **Cheap falsification gates before expensive work.** For any hypothesized fix, build the smallest
  inference-only probe, set a numeric go/no-go gate, and run it (~1 machine-hour) *before* training
  or big infra. We killed multiple bad hypotheses this way for near-zero cost.
- **Adversarial verification of every finding.** Before believing a positive result, try hard to
  *refute* it (independent re-derivation, different metric, different seed). Before believing a
  negative, confirm the baseline reproduces exactly. (Our selector's baseline reproduced a prior run
  to the decimal — that's what made the negative trustworthy.)
- **Model-independent metrics where possible.** We measured retrieval *ranking* (independent of the
  downstream label discretization) and only used end-task solve as corroboration. In RAG: report
  retrieval FullCov/rank *and* end-answer accuracy separately; don't let generation confound
  retrieval claims.
- **Significance, not vibes.** Per-query paired tests (sign test / McNemar on win/loss counts),
  not just aggregate deltas. A 2-point aggregate move is often noise; a per-query
  350-vs-6 win split at p<10⁻³⁰ is not.
- **Honest negatives as contributions.** "We tested X, Y, Z and they don't work, here's the
  mechanism" is a *stronger* deliverable than a hand-waved "future work," and it forecloses the
  obvious objections.
- **Reproducibility bundle: every headline number → one script + one CSV.** We kept a
  `CANONICAL_SOURCES.md` mapping each paper number to its exact source, and a
  `HEADLINE_NUMBERS.csv` mirror. If a number isn't reproducible from its listed source, it's a bug.
  Do this from day one — it's the cheapest credibility you can buy.
- **Don't overclaim the learned component.** Our biggest early error was framing it as "learned
  retrieval wins." Reframing to "retriever-agnostic system; the right retriever depends on the
  regime" was both more honest and more defensible. In RAG: resist "our fancy retriever beats
  BM25" — it probably doesn't, everywhere.

---

## 5. What does NOT transfer cleanly (caveats)

- **The reader's soundness.** Jigsaw's verifier is *exact* — a returned match is provably correct,
  so the only failure is coverage. An **LLM reader is not sound**: it can hallucinate even with the
  right context. So in RAG you need **both** coverage metrics **and** answer-grounding/faithfulness
  checks (citation verification, NLI-entailment against retrieved text).
- **"Labels" → "terms/entities."** Jigsaw's selectivity was over discrete vertex labels. The RAG
  analog is lexical/term/entity selectivity (IDF, doc-frequency, entity cardinality) — same shape,
  different unit. Validate the analog holds on your corpus before trusting the routing rule.
- **Structure availability.** Jigsaw had an explicit graph to expand over. RAG expansion needs a
  structure — chunk adjacency, document hierarchy, or an entity/knowledge graph. If you don't have
  one, §3.5 expansion is weaker (and the GraphRAG-style investment may be the lever).
- **Exactness of the objective.** FullCov assumed a well-defined set of "required" items (the
  planted match). In RAG, "required passages" come from gold annotations or distant supervision and
  are noisier — budget for annotation quality when you compute FullCov.
- **The diffuse-query wall may be inherent.** Our foreclosure suggests some multi-hop/aggregation
  retrieval is *hard for any embedding-only method* at scale. Don't assume a retrain will fix it;
  the honest fallback may be a different index (graph/agentic multi-step retrieval) or accepting a
  bounded miss rate.

---

## 6. A concrete starting action list for the RAG project

1. **Build the eval harness first:** bucket queries into single-passage / multi-passage /
   multi-hop-aggregation; compute **FullCov@k**, recall@k, and end-answer accuracy **per bucket**.
2. **Establish the two baselines honestly:** BM25/lexical and dense — measured on the *same*
   queries, per bucket. Expect a crossover.
3. **Compute the offline selectivity signal** (query IDF / entity doc-frequency) and test a
   **selector**: route by selectivity vs always-dense vs always-BM25 vs hybrid-fuse. Report per
   bucket with a per-query significance test.
4. **Before any fancy fix, run the cheap falsification probe** (§3.3) for whichever trick is
   tempting (HyDE / decomposition / ColBERT / RAG-fusion), with a numeric gate.
5. **Add expansion + lossless pre-filters** (§3.5); attribute every failure to coverage vs
   generation.
6. **If diffuse-query FullCov is the wall,** decide deliberately between retriever fine-tuning
   (coverage/min-CVaR loss), a graph/entity index, or agentic multi-step retrieval — don't patch it
   with prompt-side query expansion.
7. **Keep a `CANONICAL_SOURCES.md` from commit #1.**

---

## 7. Evidence appendix (the Jigsaw numbers behind each claim)

| Claim | Evidence |
|---|---|
| No universal retriever; route by selectivity | 3-dataset crossover — near-unique: FeatureIndex beats learned (Cora rank 2 vs 5, Arxiv 4 vs 44, MAG 96.7% vs 88.6%); coarse: learned wins (Arxiv 44 vs 104), FeatureIndex→random (104/200). Sign test p<10⁻³⁰ both regimes. |
| Selectivity is an offline signal | median partitions/label: Cora feature 1 / class 12; Arxiv feature 1 / class 124.5 — monotone with the crossover. |
| Coverage objective helps | FullCov@100 66%→82% (locked Arxiv, controlled ablation, McNemar p=4.0×10⁻¹⁰). |
| Inference-time tricks don't fix representation gaps | 6 remedies (multi-vector MaxSim, fine parent, fine overlap 1/2-hop, dual-granularity late interaction, coarse diffusion): best +2.3 vs +20 gate; stitch −8 to −13. Mechanism: degree 948/2000 coarse, 568/10000 fine. |
| Difficulty ∝ evidence dispersion | footprint→coverage monotone: 50 nodes/partition→98% FullCov; 1.7 nodes/partition→9%. |
| Coverage is the failure mode | direct exact solve 0/15 (infeasible); with retrieval, 88.6% — misses are retrieval misses, verifier is exact. |
| Bounded-memory serving | 2.4 GB streaming vs 10.2 GB whole-graph residence (4.2×). |

*Numbers are from the Jigsaw benchmark (Cora 19.8K / OGBN-Arxiv 169K / OGBN-MAG 1.9M nodes). They
are evidence for the transferable principles, not targets for the RAG project — re-measure on your
corpus.*
