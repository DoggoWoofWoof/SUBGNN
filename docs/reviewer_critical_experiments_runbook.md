# Reviewer-Critical Experiments — Runbook

These four experiments are the highest-leverage acceptance boosters identified by the
2026-07-11 audit (all three venue reviewers independently flagged them). They are the
things a text edit **cannot** fix — each needs the MAG/Arxiv data + deployed models on a
compute box. Ordered by leverage. Each entry says exactly what to run, which existing
script to extend, and the expected artifact so the paper can cite it.

Deployed model: `models/mag-6_layer-model-rgcn_seed7203_walkaware_best_fullcov_DEPLOYED.pth`.
Harness entry points: `scripts/lightning_production_benchmark.py`,
`scripts/benchmark_retrieval.py`, `scripts/retrieval_strategies.py`,
`scripts/benchmark_non_neural.py`, `scripts/probe_selective_overlap.py`.

---

## 1. Coarse-label sweep of Jigsaw itself  (THE kill-shot fix)

**Why:** the paper shows classical matchers collapse at coarse labels (12/27) but never runs
Jigsaw there. Its own no-exact-label ablation (MAG 88.9→52.8%) implies Jigsaw degrades too.
This asymmetry is currently indefensible. A DB reviewer will reject on it.

**Run:** re-run the MAG production matrix at decreasing label selectivity by widening the
hash modulus in the label discretizer: `sigma = md5(x) mod M` for `M ∈ {1e6, 1e4, 1e2, 10}`
plus the raw class label (39 classes). The discretizer lives where `mag_type_rel_v1` features
are built (`src/data.py`) / the export to `vertexlabelledlad`; parameterize `M` and re-emit
labels only (no re-partition, no re-train — retrieval is unchanged; only pruning + verifier
selectivity change).

**Report:** solve rate, candidate size, timeouts, peak RSS at each `M`, for Jigsaw vs FilterAll
vs one classical matcher on the *same* labels. Expected story: Jigsaw's advantage over classical
*widens* as labels coarsen (retrieval still localizes; classical candidate explodes) — if it
holds, this becomes a headline figure. If Jigsaw also collapses, report it honestly and scope
the contribution to label-rich regimes.

**Output:** `runs/coarse_label_sweep_mag/results/*_per_query.csv` → new table + figure.

---

## 2. Unlearned label→partition inverted-index baseline

**Why:** with near-unique md5 labels, an inverted index (query label → partitions containing it
→ local expansion → Glasgow) may match Jigsaw with no GNN/FAISS. Its absence is conspicuous;
it is the natural cheap competitor.

**Run:** add a policy `inverted_index` alongside the six in `scripts/retrieval_strategies.py`:
build `label → {coarse partition ids}` from the cached partition label sets (offline, one pass),
then per query rank partitions by count of shared query labels. Feed the same cascade
(overlap → prune → verify). No training.

**Report:** add its row to the production matrix (all three datasets) and to the MAG significance
table (paired McNemar vs Jigsaw). Two acceptable outcomes: (a) it ties Jigsaw at high selectivity →
reframe the learned-retriever case around coarse labels (ties into Exp. 1); (b) it loses →
the learned retriever's necessity is *strengthened*. Either way the objection is answered.

**Output:** `runs/inverted_index_baseline/` + one new row in `HEADLINE_NUMBERS.csv`.

---

## 3. Memory-capped streaming FilterAll vs streaming Jigsaw (matched RSS)

**Why:** Table 2 shows FilterAll beats Jigsaw on MAG on BOTH accuracy (98.4 vs 88.6) AND
end-to-end latency (5.85 vs 6.97 s). The text calls FilterAll "not scalable" — contradicted by
its own numbers. The learned retriever's value must be pinned to a *measured* axis.

**Run:** extend the streaming serve (`scripts/streaming_serve_smoke.py` / the 54-query grid in
`runs/lightning_completion/mag_streaming_grid_v5/`) to also run FilterAll under the SAME
resident-partition cap (8 partitions). Then run both classical CFL/DP-iso/GQL under a hard
cgroup memory limit (e.g. `systemd-run --scale MemoryMax=4G` or a container `--memory=4g`) on
full MAG.

**Report:** at a matched 2.4 GB / 4 GB cap: does streaming FilterAll OOM or thrash while
streaming Jigsaw sustains ~88%? Do the resident classical matchers OOM at 4 GB? This turns the
memory frontier from one asserted sentence into the headline figure and neutralizes both the
"FilterAll dominates" and "10 GB isn't a barrier" attacks.

**Output:** `runs/mag_memcap_headtohead/` → replaces/augments `fig_memory_latency.png` panel.

---

## 4. Fresh-seed evaluation of the deployed walk-aware encoder

**Why:** the walk-aware encoder was retrained *after* observing walk/multi-coarse failures on the
same families/seeds now reported (20260607/08) — reads as adaptive test-set fitting.

**Run:** generate one new query seed (e.g. 20260609) with the same
`scripts/lightning_production_benchmark.py` generator, all eight families / three sizes, and
evaluate the *frozen* deployed model — no retrain, no tuning.

**Report:** MAG per-family + overall solve rate on the held-out seed alongside the deployed
88.6%. If within noise, the retrain generalized (kills the objection); if it drops on
walk/multi-coarse, report the honest generalization gap.

**Output:** `runs/mag_freshseed_20260609/` → one column added to the family table.

---

## Also cheap & text-only (already applied in the papers, listed for completeness)
- Out-of-core/distributed prior art cited + differentiated (DUALSIM, STwig/Trinity, G-thinker).
- "0/15 infeasible" rescoped to "the CP-based Glasgow solver" at abstract + contributions.
- md5 label hashing scoped to hashed-label semantics ("near-unique" → collision-honest).
- Offline-cost / amortization limitation added.

## Priority if compute is scarce
Do **1** and **3** first — they convert the two attacks that all three reviewers rated as
rejection-grade. **2** is one afternoon of coding. **4** is the cheapest (one benchmark run,
no new code).
