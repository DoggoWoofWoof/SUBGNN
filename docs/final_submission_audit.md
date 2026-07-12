# Final Submission Audit

Date: 2026-06-09

This is the strict acceptance-readiness audit for the MLG submission. It
separates paper-ready evidence from active experiments and from claims that must
not appear in the final manuscript.

## Current Verdict

Jigsaw has a workshop-worthy core contribution:

- retrieval-constrained exact verification is a clear and useful formulation;
- FullCov is the correct necessary retrieval metric for downstream exact
  verification;
- the matched-budget final objective significantly improves Arxiv K-hop
  FullCov over the matched control;
- the experiments expose meaningful failure regimes rather than hiding them.

The paper is submission-ready for the MLG workshop track, with one important
positioning constraint: it should be sold as a workshop contribution with a
strong Arxiv retrieval/ablation story, not as a broadly solved industrial-scale
subgraph matching system. The GraphSAGE baseline and GraphSAGE+FullCov run are
complete: the result strengthens the FullCov objective story but rules out any
claim that the residual GIN encoder alone is responsible for the gain. A
positive third-dataset result is still missing.

## Paper-Ready Evidence

| Question | Evidence | Result |
| --- | --- | --- |
| Does the final objective improve retrieval? | `runs/logs/fair_ablation_locked_aggregate.md` | FullCov@20/50/100 improves from `21.8/39.0/66.0%` to `26.0/48.8/82.0%`; paired exact p-values `0.0046/9.8e-6/4.0e-10`. |
| Does it beat simple retrieval heuristics on the same queries? | `benchmarks/paper_results/diagnostics/results_arxiv_non_neural_locked_q20.csv` | At K=50, final Jigsaw `48.8%` vs Mean Feature and Neighbor Expansion `37.5%`. |
| Does it beat a learned GraphSAGE encoder baseline? | `runs/logs/graphsage_standard_locked_aggregate.md`, `runs/logs/graphsage_fullcov_locked_aggregate.md` | Standard GraphSAGE final reaches `26.0/47.0/68.5%`; GraphSAGE+FullCov reaches `28.5/52.0/80.0%`; Jigsaw-GIN FullCov reaches `26.0/48.8/82.0%`. The objective transfers, and GraphSAGE+FullCov is the strongest K=50 encoder variant. |
| How does query structure affect retrieval? | `benchmarks/paper_results/diagnostics/eval_arxiv_20.txt`, `eval_arxiv_50.txt`, `eval_arxiv_100.txt` | Single and Multi-Fine are near-perfect; K-hop degrades from `49%` FullCov@50 at 20 nodes to `15%` at 100 nodes. |
| Does retrieval help exact verification? | `benchmarks/paper_results/glasgow_benchmark_corafull_all.csv` | Attributed CORA compact queries achieve `3.9-25.8x` end-to-end speedup; K-hop is a `0.8x` slowdown. |
| What is fixed neural retrieval latency? | `runs/logs/retrieval_arxiv_khop_q100_sizes20_seed20260607_timed_summary.csv` and seed `20260608` | Query encoding plus full FAISS coarse ranking averages about `5.7-5.9 ms/query`; model/index build is one-time and excluded. |
| Does the same recipe scale to million-node heterogeneous graphs? | `runs/logs/mag_retrieval_locked_aggregate.md` | No. Best budget-100 FullCov is `16/200`, `2/200`, and `0/200` for target sizes 20, 50, and 100. |

## Reviewer Concern Mapping

### Missing GNN Encoder Baseline

Status: **complete enough for submission; include in the paper**.

Completed run: `graphsage_contrastive_arxiv_seed7101`

- GraphSAGE encoder;
- standard hierarchical InfoNCE only;
- no FullCov objective, top-K barrier, CVaR, or live positive re-encoding;
- same 9,000 optimizer steps, query generator, batch size, optimizer, and
  scheduler as the fair Jigsaw protocol.

The run completed 90 epochs / 9,000 optimizer steps with no OOM or traceback.
Standard contrastive loss decreased from `1.5113` at epoch 1 to `0.1530` at
epoch 90. The best validation checkpoint was selected at epoch 65.

Locked retrieval on seeds `20260607` and `20260608` gives:

| Checkpoint | FullCov@20 | FullCov@50 | FullCov@100 | Recall@20 | Recall@50 | Recall@100 | Avg max true rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphSAGE best | 52/200 (26.0%) | 88/200 (44.0%) | 139/200 (69.5%) | 0.7008 | 0.8399 | 0.9395 | 72.50 |
| GraphSAGE final | 52/200 (26.0%) | 94/200 (47.0%) | 137/200 (68.5%) | 0.7082 | 0.8466 | 0.9406 | 73.08 |
| GraphSAGE+FullCov best | 57/200 (28.5%) | 104/200 (52.0%) | 160/200 (80.0%) | 0.7117 | 0.8714 | 0.9621 | 58.60 |
| GraphSAGE+FullCov final | 56/200 (28.0%) | 103/200 (51.5%) | 162/200 (81.0%) | 0.7105 | 0.8709 | 0.9670 | 57.69 |
| Jigsaw-GIN final objective | 104/400 (26.0%) | 195/400 (48.8%) | 328/400 (82.0%) | 0.7012 | 0.8740 | 0.9676 | 58.97 |

Interpretation: GraphSAGE is a strong and necessary learned baseline. More
importantly, GraphSAGE+FullCov improves over standard contrastive GraphSAGE on
the same locked seeds and exceeds the two-seed Jigsaw-GIN aggregate at K=20 and
K=50. Jigsaw-GIN remains slightly stronger at K=100. Therefore the manuscript
must frame the contribution as FullCov-aligned retrieval and exact verification
with encoder-agnostic evidence, not as a residual-GIN architecture win.

Submission-day note: the first Spanish GraphSAGE+FullCov Arxiv run was canceled
around epoch 42 without OOM or traceback. It was relaunched in the same
workspace with `-Resume`, verified to load
`/cache/arxiv_graphsage_final_loss_arxiv_seed7101_checkpoint.pth`, and finished
90 epochs. Locked retrieval completed on seeds `20260607` and `20260608`.

### Only Two Positive Datasets

Status: **partially unresolved**.

- Flickr (89K nodes, 100 coarse partitions) was wired and smoke-tested
  successfully, but a fair 9,000-step run is estimated at roughly 15 hours and
  is not submission-feasible.
- Physics (34K nodes, 35 coarse partitions) and Flickr (89K nodes, 100 coarse
  partitions) were wired and smoke-tested. Both fair 9,000-step runs are
  estimated at roughly 15 hours. Physics is additionally weak evidence because
  K=50/100 retrieves its entire coarse partition set.
- MAG remains a large-scale negative diagnostic, not a positive result.

Do not present a rushed or under-trained Flickr result as comparable evidence.

### Low Arxiv FullCov@50

Status: **addressed by scope, not solved**.

The correct claim is that the final objective raises FullCov@50 from 39.0% to
48.8% under matched budget. This is a significant improvement, but it still
misses a required partition for slightly over half of hard K-hop queries.
Real-world utility claims must emphasize compact patterns and treat broad
K-hop retrieval as the main remaining limitation.

### MAG Failure

Status: **addressed honestly**.

The manuscript attributes the failure to the combination of:

- relation information lost by homogeneous conversion;
- sparse live-positive updates relative to 2,000 coarse partitions;
- heterogeneous high-degree neighborhoods fragmented by topology-only METIS;
- query footprints growing to 26.1 and 54.6 true partitions for 50- and
  100-node K-hop queries.

These explanations are plausible diagnostics, not proven causal conclusions.
The strongest local diagnostic is the worst-positive rank distribution from the
locked MAG retrieval CSVs. For target sizes 20/50/100, the median rank of the
last required coarse partition is roughly `1296/1733/1879` out of 2,000. Even
an oracle policy that simply retrieves every partition up to that worst-positive
rank would achieve only `7/200`, `2/200`, and `0/200` FullCov by rank 100. This
means MAG is not failing because of a small amount of noisy stitching after a
good seed; the ranker is usually placing at least one required partition in the
bottom half of the index.

### Wall-Clock Latency and Offline Cost

Status: **addressed with explicit boundaries**.

- CORA table reports end-to-end time including query embedding, retrieval,
  candidate assembly, and Glasgow.
- Independent Arxiv timing runs measure neural retrieval at about 5.8 ms/query.
- The one-time model/index build is excluded from per-query latency and must be
  disclosed.
- Arxiv training takes roughly 2-3 hours per 9,000-step run in the observed
  logs. The amortization claim applies only to relatively static,
  high-query-volume graphs.

## Methodological Caveats That Must Stay Visible

1. CORA Glasgow uses deterministic hashes of node features as strict vertex
   labels. Approximately 97.6% of CORA nodes have unique labels. The CORA
   result is therefore an attributed/planted-match timing benchmark, not
   evidence of acceleration on hard unlabeled structural instances.
2. Glasgow is exact only inside the retrieved candidate graph. Jigsaw is not
   globally complete when FullCov fails.
3. The conditional Arxiv Glasgow diagnostic uses FullCov gating and label
   pruning. It demonstrates verifier feasibility after successful retrieval;
   it is not an unbiased end-to-end success rate.
4. Dynamic boundary expansion does not beat fixed neural ranking in the locked
   final comparison and should remain secondary or omitted.
5. The component ablations do not improve monotonically. CVaR and the barrier
   help only when combined with live positive re-encoding in the final
   objective.

## Active Experiments

| Experiment | Purpose | Decision Rule |
| --- | --- | --- |
| Arxiv GraphSAGE+FullCov objective | Test whether the final loss transfers to GraphSAGE | Completed; include as encoder-transfer evidence. |
| MAG GraphSAGE+FullCov diagnostic | Test whether GraphSAGE rescues MAG scale failure | Still not paper-positive. Latest seen validation trailed completed MAG GIN. Focus shifted to overlap/pruning cascade. |
| MAG overlap cascade | Test whether retrieval + overlap + exact pruning can make MAG verifier-sized | Verified on 100 locked 20-node K-hop queries: fixed `49/100`, hybrid `54/100` solved with budgets `20/50/100`; hybrid average final label-pruned candidate `21.7k` nodes. Include only as partial scalability diagnostic. |
| Arxiv fixed cascade | Verify covered candidates at practical budgets | Verified: `99/100` solved on 20-node K-hop, with `85/100` already solved at K=20; avg solver `24.7ms`. Strong paper support. |
| Cora fixed cascade | Existing-model sanity check, no retrain | Verified: `100/100` solved for 20/50/100-node queries at K=20. Timing/sanity only because Cora has 20 coarse partitions. |
| Physics/Flickr smoke tests | Test feasibility of a third positive dataset | Do not present smoke results as comparable evidence; a fair full run is not submission-feasible. |
| Rank-neighbor / prefix-seeded stitching retrieval | Test whether high-precision top-1/2/5/10 anchors improve dynamic expansion | Completed on GraphSAGE+FullCov, seeds `20260607/20260608`, query sizes `20/50/100`. Keep internal/future-work only: high-precision anchors slightly help B=50, but the gain is too small for the main paper. |

## Rank-Neighbor Stitching Probe

The June 9 stitching probe tested a deterministic retrieval rule proposed as:
take a high-ranked seed set, expand only through top-100 ranked neighbor
partitions in the coarse partition graph, and optionally use query/partition
mean-feature similarity as a secondary rank signal. It was evaluated without
labels, oracle truth, or Glasgow on the Spanish workspace using the
GraphSAGE+FullCov Arxiv best checkpoint, two locked seeds, and 100 K-hop
queries per target size.

Aggregate artifact:

- `runs/logs/retrieval_arxiv_khop_stitch_v1_graphsage_q100_sizes20_50_100_aggregate.md`
- `runs/logs/retrieval_arxiv_khop_stitch_v1_graphsage_q100_sizes20_50_100_aggregate_summary.csv`

Result: do not add it to the main paper. At target query size 20, stitching
improves the K=20 bucket from fixed `57/200` to `62/200` FullCov, but at K=50
it only matches the best old dynamic method (`106/200`), and at K=100 it trails
the best old dynamic method (`164/200` versus `167/200`). At target size 50,
the best old dynamic method remains better at K=50 (`54/200` versus `51/200`)
and K=100 (`126/200` versus `124/200`). At target size 100, stitching only
matches the old dynamic method at K=50 and K=100 (`24/200` and `85/200`).
Mean-feature retrieval is weaker than the learned ranking at all fixed
headline budgets. The useful conclusion is that neighborhood consistency can
slightly clean very small buckets, but it is not a submission-strength result.

Follow-up prefix-seeded probe: the precision idea is useful specifically as a
dynamic expansion seed. We therefore forced the first `k=1/2/5/10` partitions
to be high-precision anchors, then expanded from those anchors with the old
hybrid neighbor rule or the stricter stitching rule. The seed precision was
indeed high: for 20-node queries the best seed precision was `0.730/0.635/
0.431/0.303` at prefix `1/2/5/10`; for 50-node queries it was `0.915/0.803/
0.622/0.472`; for 100-node queries it was `0.935/0.900/0.772/0.654`.

That confirms the mechanism but not a paper-worthy gain. At B=50, prefix-seeded
hybrid improves over the old dynamic selector by only `+2/200`, `+1/200`, and
`+1/200` FullCov for target query sizes `20/50/100` respectively:
`106 -> 108`, `54 -> 55`, and `24 -> 25`. At B=100 it ties or slightly trails
the old dynamic selector. The stricter prefix-seeded stitching variant usually
ties or trails prefix-seeded hybrid. Artifact:

- `runs/logs/retrieval_arxiv_khop_prefix_seed_v1_graphsage_q100_sizes20_50_100_aggregate.md`
- `runs/logs/retrieval_arxiv_khop_prefix_seed_v1_graphsage_q100_sizes20_50_100_aggregate_summary.csv`

## Final Paper Checklist

- Insert the completed GraphSAGE and GraphSAGE+FullCov rows and describe the
  comparison without architecture-only claims.
- Add Physics only if the full run and retrieval evaluation are complete and
  non-trivial.
- Keep MAG as a scalability diagnostic: retrieval-only is negative, but fixed overlap/pruning cascade partially rescues 20-node queries.
- Report fixed FullCov as the primary metric; keep recall secondary.
- Report query size/type breakdown and impossible-at-K counts.
- Keep the Cora label-uniqueness caveat and K-hop slowdown visible.
- Compile twice, visually inspect every page, and confirm main text remains
  within the 12-page limit. Latest checked artifact:
  `paper/samplepaper.pdf`, rendered to `runs/paper_render_20260609_final/`.
- Do not submit root scratch outputs; use `paper/samplepaper.tex` and
  `paper/samplepaper.pdf` as the canonical manuscript artifacts.

## Submission-Day Decisions

- The method text was rewritten around a generic encoder interface
  `f_theta(G) -> z_G`. Residual GIN is now framed as the primary
  matched-budget instantiation, while GraphSAGE is the encoder-transfer check.
  This keeps the contribution on FullCov-aligned retrieval plus exact
  verification rather than on a single backbone.
- Dynamic boundary expansion was removed from the main paper because it did
  not improve fixed neural ranking and distracted from the stronger objective
  and scope story.
- The main paper explicitly acknowledges that a successful third medium-scale
  graph remains missing; Physics/Flickr smoke tests are not presented as
  comparable evidence.
- MAG is framed as unresolved at broad query sizes, with one verified partial
  20-node cascade result. Retrieval-only remains weak.
- Final source edits were recompiled twice with MiKTeX and rendered with
  PyMuPDF to `runs/paper_render_20260610_final/`. The PDF has 13 pages total;
  main text ends on page 12 and references begin on page 12.

## Final Brutal Verdict

Acceptable workshop paper if the submission stays honest. The strongest story
is not "Jigsaw beats every GNN"; it is:

- FullCov is the correct retrieval condition for exact verification.
- The FullCov-aligned objective improves matched-budget retrieval on hard Arxiv
  K-hop queries.
- The gain transfers from GIN to GraphSAGE, so the contribution is objective-
  and system-level.
- Exact verification is preserved only inside covered candidates.

The weakest points remain:

- no successful third positive medium/large dataset;
- Arxiv FullCov@50 is improved but still only about half of hard K-hop queries;
- MAG retrieval-only is negative; the 20-node overlap cascade is promising
  (`54/100`) but not a broad scalability win;
- CORA speedups are on an attributed/planted timing benchmark with mostly
  unique labels.

For ECML PKDD MLG, this holds water because those weaknesses are disclosed and
the paper now includes the requested learned encoder baseline.
