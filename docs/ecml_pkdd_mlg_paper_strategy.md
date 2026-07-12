# ECML PKDD MLG Paper Strategy

This document is the critical submission plan for the June 9 ECML PKDD / MLG
version. It translates the current experiments into the strongest defensible
paper story and lists the paper edits that should happen before submission.

## Venue Fit

The MLG call explicitly welcomes graph algorithms, graph representation
learning, ML for combinatorial problems on graphs, benchmarking,
reproducibility, libraries/tools, and empirical studies. Jigsaw fits best as a
systems-and-benchmarking paper for **learning to retrieve candidate regions for
exact subgraph matching**, not as a claim that a GNN has solved global subgraph
isomorphism.

The paper should therefore read like:

> Exact subgraph solvers are reliable but hard to scale. Jigsaw learns a
> partition-level retriever whose objective is aligned with the condition an
> exact verifier needs: every true partition must be present. The verifier is
> exact inside the retrieved candidate region.

That framing is much stronger than the old "fast approximate retrieval plus
high FSA/BSA" framing because it directly addresses the real bottleneck.

## Strongest Defensible Claims

Use these as the paper's main claims:

1. **FullCov is the right retrieval metric for exact verification.** Recall can
   be high while exact verification is impossible because one missed partition
   can remove the true match.
2. **The final matched-budget objective improves neural retrieval.** On locked
   Arxiv K-hop held-out queries, the final objective significantly improves
   fixed FullCov@20/50/100 over the matched control under the same 9,000
   optimizer steps. The K=50 result is especially important because it is the
   practical midpoint budget: FullCov rises from 39.0% to 48.8% with paired
   exact \(p=9.8\times10^{-6}\).
3. **Dynamic retrieval is optional secondary evidence.** It is useful as an
   engineering probe, but it does not currently produce the cleanest improvement
   story. Do not foreground its non-significant objective gap in the main paper.
   If space is tight, keep only a compact system-budget row and move the
   detailed dynamic selector discussion to an appendix.
4. **Exactness is conditional on retrieval coverage.** Glasgow remains exact,
   but only over the stitched candidate graph. The paper must not imply global
   exactness unless the candidate region contains all true match partitions.
5. **MAG is a scalability stress test, not a headline success.**
   The completed retrieval-only rerun is weak: best dynamic FullCov is only
   `16/200` for 20-node queries, `2/200` for 50-node queries, and `0/200` for
   100-node queries at budget 100. Use MAG only as an honest scalability
   diagnostic unless a future objective materially changes this.

## June 9 Evidence Corrections

- The matched non-neural comparison now uses the exact locked 200-query Arxiv
  suite. At K=50, final Jigsaw reaches `48.8%` FullCov versus `37.5%` for both
  Mean Feature and Neighbor Expansion. This is a meaningful gain, not a
  doubling.
- Independent Arxiv timing runs instrumented in
  `scripts/benchmark_retrieval.py` measure query encoding plus full coarse
  FAISS ranking at about `5.7-5.9 ms/query`. This excludes one-time model and
  index construction.
- CORA exact-verification timing uses strict hashed feature labels; about
  `97.6%` of CORA nodes have unique labels. Present it as an attributed,
  planted-match wall-clock sanity check, not a hard unlabeled-matching result.
- A fair GraphSAGE standard-contrastive baseline and a GraphSAGE+FullCov run
  are complete. Use them to show that FullCov alignment transfers across
  encoders, not that residual GIN alone is responsible.
- Flickr/Yelp are not submission-feasible at the fair 9,000-step budget.
  Physics/Flickr smoke tests should not be presented as comparable evidence.
  MAG remains the large-scale negative diagnostic.

## Unsafe Claims To Remove

Remove or rewrite these everywhere in the manuscript:

- "finds all subgraph isomorphisms" without the qualifier "inside the retrieved
  candidate region";
- "near-perfect accuracy" as a primary headline, because FSA/BSA condition on
  solver success and hide retrieval misses;
- "K-hop failure is purely capacity-saturated, not capability-limited" as a
  broad claim. Say instead that K-hop exposes both top-K capacity and neural
  ranking difficulty;
- "dynamic retrieval solves K-hop" unless FullCov is actually complete at the
  chosen budget. If dynamic retrieval is not improving the headline metric, do
  not burn main-paper space explaining that failure;
- "label pruning" as part of neural retrieval. Label pruning is verifier-side
  diagnostic machinery;
- comparing a continuation checkpoint against a from-scratch control as causal
  evidence;
- MAG success claims; the completed retrieval-only CSVs support only a
  scalability limitation/diagnostic claim.

## Reviewer Concern Mapping

| Concern | Paper response |
| --- | --- |
| Benchmark clarity | Make FullCov@K the primary metric; report recall only as secondary. Add query true-partition count, impossible-at-K, and max true-partition rank. |
| Exactness ambiguity | State that Glasgow is exact inside the retrieved candidate. Separate retrieval selection from exact verification. |
| Weak/unclear ablations | Add the fair matched-budget objective ablation: control, CVaR-only, CVaR+top-K barrier, final. Use fixed 9,000 optimizer steps. |
| Solver confounds | Select models using retrieval-only FullCov. Use Glasgow only in a small end-to-end table after retrieval is fixed. |
| K-hop failures | Present K-hop as a stress test. Report FullCov and true partition footprint by query size. Do not hide failures behind average recall. |
| Dataset breadth | Keep Cora exact-verification speedup, Arxiv retrieval and objective ablation, and add MAG only as a diagnostic unless complete rerun results are strong. Use PubMed/Physics if a positive third dataset is needed. |
| Reproducibility | List fixed seeds, training steps, checkpoint selection, query generation, retrieval budgets, and Modal/local artifact paths. |
| Exploratory methods | Move MC dropout, MCTS-style probes, fine-coverage loss, node loss, and multiview reranking to negative ablations or omit. |

Yelp is explicitly excluded from the June 9 evidence package. A smoke run in
the `darkphoenix696969696969` Modal workspace used the same cached-negative
cross-dataset training path but reached only step `6/10` of epoch 1 after about
`15m52s`; the run was stopped. Do not mention Yelp except, at most, as an
internal discarded attempt.

## Main Paper Structure

### Abstract

Replace the current abstract with a FullCov-aware abstract:

- one sentence on exact subgraph matching being bottlenecked by candidate
  domains;
- one sentence on Jigsaw learning partition retrieval before exact verification;
- one sentence defining FullCov as the necessary retrieval condition;
- one sentence with matched-budget Arxiv gains;
- one sentence with Cora speedup/exact verification;
- one sentence acknowledging K-hop/MAG as scalability stress tests.

Do not lead with FSA/BSA. Those are secondary once retrieval succeeds.

### Contributions

Rewrite contributions to:

1. a retrieval-constrained exact verification framework;
2. a FullCov-aligned multi-positive partition objective;
3. a fixed retrieval benchmark protocol across K=20/50/100 that separates
   retrieval coverage from solver success;
4. an empirical study with matched-budget ablations and query-type stress tests;
5. a reproducible implementation and artifact manifest.

### Method

Keep the residual GIN encoder description, but update training to match the
final objective:

\[
\mathcal{L}
=
0.2\mathcal{L}_{fine\text{-}NCE}
+
0.8\mathcal{L}_{coarse\text{-}NCE}
+
\mathcal{L}_{coverage}.
\]

The coverage objective must include:

- all-positive partition coverage;
- CVaR over the weakest true partitions;
- the dynamic top-K barrier: K=20, then 30, 40, ... when a query touches more
  partitions;
- live re-encoding of up to 24 true positive partitions.

Refer to `docs/jigsaw_loss_and_retrieval_method.md` for the longer derivation
and worked example.

### Experiments

Reorder experiments around the question reviewers care about:

1. Does the retriever cover all required partitions?
2. Does the final loss improve coverage under a fair training budget?
3. Does the improvement persist across query sizes, query types, and datasets?
4. Once coverage is achieved, does exact verification work and how fast?
5. Which stress cases remain hard, and why?

## Required Main-Paper Tables

The paper should not look Cora-specific. Cora is the small-graph exact-solver
speedup sanity check, while Arxiv is the main hard retrieval benchmark. The
main text should therefore report Arxiv fixed FullCov at K=20/50/100, then a
small Arxiv Glasgow-style verification table showing what happens once a
candidate region is retrieved.

## Benchmarks Beyond Glasgow

Yes, we need benchmarks beyond Glasgow, but the additional benchmarks should
test **retrieval selection**, not just swap in more exact solvers. Glasgow is
the downstream verifier; it cannot diagnose the learned model if the candidate
region is missing a required partition.

Main-paper retrieval baselines should be:

- **Random top-K partitions:** sanity lower bound for FullCov@K.
- **Graph-neighbor expansion without learning:** start from a simple anchor or
  METIS-local seed and expand over the coarse partition graph. This tests
  whether boundary expansion alone explains the result.
- **Feature/degree/PageRank-style heuristic ranking:** cheap non-neural
  baselines that test whether coarse structural popularity is enough.
- **Learned Jigsaw fixed top-K:** the primary model-quality result.
- **Optional locked dynamic retrieval:** system-level result only if space
  permits; do not make it the core claim because fixed retrieval gives the
  cleaner causal objective result.

Exact-solver baselines should be limited:

- **Full-graph Glasgow on Cora:** feasible exact baseline for speed and
  correctness.
- **Jigsaw + Glasgow on a small FullCov-gated subset:** demonstrates that exact
  verification succeeds once retrieval covers the target.
- **Other exact solvers (VF2/RI/DPISO/CFL/TurboISO):** optional appendix sanity
  checks only. They do not replace retrieval FullCov because they all fail when
  retrieval omits a true partition.

This is the most reviewer-aligned benchmark framing: compare learning against
retrieval baselines, then use Glasgow to verify exactness after candidate
coverage is achieved.

## Metric Package

For every dataset where we report retrieval, use the same metric package:

- FullCov@20/50/100 as the primary metric;
- recall@20/50/100 as a secondary diagnostic;
- average and maximum true coarse partitions per query;
- impossible-at-K count;
- maximum true-partition rank;
- query graph target sizes 20/50/100;
- query-type breakdown when available: single, multi-fine, multi-coarse, k-hop.

This directly addresses the reviewer requests for stronger metrics, ablations,
different K values, different query sizes, and broader benchmark coverage. It
also keeps the story consistent across Cora, Arxiv, and any larger dataset.

### Table 1: Matched-Budget Objective Ablation

Use locked Arxiv K-hop held-out seeds `20260607` and `20260608`.

| Result | Control | Final objective | Claim |
| --- | ---: | ---: | --- |
| Fixed FullCov@20 | 87/400 (21.8%) | 104/400 (26.0%) | +4.25 pp, exact paired p=0.0046 |
| Fixed FullCov@50 | 156/400 (39.0%) | 195/400 (48.8%) | +9.75 pp, exact paired p=9.8e-06 |
| Fixed FullCov@100 | 264/400 (66.0%) | 328/400 (82.0%) | +16.0 pp, exact paired p=4.0e-10 |

Also include CVaR-only and CVaR+top-K barrier rows, but interpret them
carefully: the component gains appear only in the combined final objective.

### Table 2: Fixed Retrieval Across K

Report fixed K=20/50/100 on the same locked Arxiv queries. This should be a
main-paper table because it gives the strongest clean result without solver or
dynamic-selector confounds.

Recommended wording:

> The final objective improves FullCov at every fixed retrieval budget, with
> the largest practically relevant gain at K=50 and the largest absolute gain
> at K=100.

### Optional Appendix Table: Dynamic Retrieval

| Result | Control | Final objective | Interpretation |
| --- | ---: | ---: | --- |
| Dynamic FullCov B=50 | 171/400 | 187/400 | useful, p=0.060 |
| Dynamic FullCov B=75 | 248/400 | 264/400 | useful, p=0.109 |
| Dynamic FullCov B=100 | 315/400 | 321/400 | similar, p=0.512 |

Do not foreground this in the main paper unless page budget permits. The table
can be used as an appendix/system diagnostic showing that boundary expansion is
reasonable, but the acceptance case should not depend on it.

### Table 3: Query Difficulty / Feasibility

For each query type and target size, report:

- average true coarse partitions;
- maximum true coarse partitions;
- impossible-at-K for K=20/50/100;
- average max true-partition rank;
- FullCov@20/50/100.

This prevents the paper from being accused of hiding K-hop difficulty.

### Table 4: Exact Verification After Retrieval

Small fixed subset only. Columns:

- retrieval method;
- candidate FullCov;
- solved count;
- timeout count;
- stitched nodes;
- Glasgow time.

This table should not be used for model selection.

Concrete Arxiv evidence already available:

| Arxiv verification setting | Candidate policy | Candidate FullCov | Solved | Solver failures after candidate FullCov | Median solver time | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Coarse top-50, V2 probe | 50 coarse partitions | 3/6 | 3/6 | 0 | 42.1s | K=50 can verify when FullCov holds, but misses half the queries. |
| Coarse top-100, V2 probe | 100 coarse partitions | 4/6 | 1/6 | 3 | 89.7s | More recall alone is not enough; unpruned candidates become too large. |
| Fine-boundary K=125, V2 q30 | FullCov-gated + label-pruned | 26/30 | 26/30 | 0 | 0.02s | Once candidate FullCov is obtained and impossible labels are pruned, Glasgow verifies all covered cases. |
| Fine-boundary K=150, V2 q30 | FullCov-gated + label-pruned | 28/30 | 28/30 | 0 | 0.02s | Higher candidate budget improves coverage without introducing solver failures after pruning. |
| Final dynamic 20->50, q30 | Fine-boundary expansion + prune | 11/30 | 11/30 | 0 | 0.00s | Practical low-budget final-model point; still under-covers many K-hop queries. |
| Final dynamic 20->100, q30 | Fine-boundary expansion + prune | 20/30 | 20/30 | 0 | 0.02s | Starting from top-20 and expanding boundaries gives a practical system point, but fixed final-objective FullCov is the cleaner causal result. |

This table is paper-usable only with careful language: it proves conditional
exact verification inside retrieved candidates. It should not be phrased as
"Arxiv full-graph Glasgow speedup", because full-graph Glasgow on Arxiv is not
a stable measurable baseline. The final-model dynamic rerun completed on June
8, 2026: B=50 achieved candidate FullCov/solved `11/30`, B=100 achieved
`20/30`, both with oracle `30/30` and zero solver failures after candidate
FullCov. Use these rows when discussing the adopted final objective.

### Table 5: Cross-Dataset Coverage

At minimum:

- Cora;
- Arxiv;
- MAG if complete final-only retrieval CSVs exist.

For MAG with 2,000 coarse partitions, the completed final-model rerun should
not be a headline table: best dynamic FullCov at budget 100 is `16/200` for
20-node queries, `2/200` for 50-node queries, and `0/200` for 100-node queries.
If included, put this in a limitations/scalability diagnostic section and use
PubMed/Physics or a homogeneous 500K--1M node graph for a positive third-dataset
claim.

## Figures To Add

1. **FullCov@K bar chart**: control vs final at K=20/50/100.
2. **Dynamic budget curve**: FullCov as B grows from 50 to 75 to 100.
3. **Max true-partition rank distribution**: show the final objective moves the
   worst required partition upward.
4. **Query footprint histogram**: true partition count by query type and size.
5. **Exact verification scatter**: stitched candidate size versus Glasgow time
   for the small verification subset.

Use simple, legible plots. No decorative figures. The paper needs evidence,
not ornament.

## Visual And Page-Limit QA

Every manuscript edit must end with a rendered PDF check, not just a TeX diff.
The MLG workshop limit is 12 pages for long papers and 8 pages for short
papers in Springer LNCS style; references, acknowledgments, and appendix do not
count according to the workshop call. Source checked on June 8, 2026:
https://mlg-europe.github.io/ and the ECML PKDD workshop-track page recommends
LNCS formatting for workshop papers:
https://ecmlpkdd.org/2026/submissions-workshop-track/. The current local
machine has MiKTeX installed, but it may not be on `PATH`. Use the absolute
binary path
`C:\Users\Swastik\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe`
from the `paper` directory. If it fails, use one of:

- Overleaf;
- a CI/render container with LNCS support.

Before submission, verify:

- total counted main-text pages are within the selected track limit;
- tables do not overflow margins;
- figures are readable in grayscale;
- captions do not create lonely lines or awkward page breaks;
- no equation, URL, or table cell protrudes into the margin;
- anonymous links and artifact references preserve double-blind constraints;
- the PDF opens cleanly and has embedded fonts.

No paper version should be called final until this visual pass is complete.

### Current Visual Pass

On June 9, 2026, after the GraphSAGE baseline and GraphSAGE+FullCov edits,
`paper/samplepaper.tex` was compiled twice locally with MiKTeX using the LNCS
class. The resulting PDF was rendered page-by-page via PyMuPDF into
`runs/paper_render_20260609_final/`.

- the PDF is 13 pages total;
- the conclusion finishes on page 12;
- references begin on page 13;
- counted main text therefore fits the 12-page long-paper limit;
- the GraphSAGE, matched-budget FullCov, query-structure, and CORA timing
  tables are visually readable and within the text margins;
- all fonts are Type1 after adding `lmodern`; no Type3 bitmap fonts remain;
- remaining LaTeX overfull-box warnings are paragraph-line warnings
  (maximum about 13.8 pt), not table or figure overflow.

The rendered artifact is `paper/samplepaper.pdf`.

## Paper Rewrite Order

1. Update abstract, contributions, and conclusion to remove unsafe exactness
   claims.
2. Replace the training objective section with the final loss.
3. Rewrite metrics around FullCov first, recall second, solver success third.
4. Add Tables 1 and 2 from the completed fair ablation, emphasizing fixed
   K=20/50/100 and especially the K=50 gain.
5. Add query difficulty/feasibility table.
6. Keep Cora end-to-end solver speedup as the compact-query sanity check.
7. Add MAG only as an honest scalability limitation; the final-only rerun is
   complete and too weak for a positive FullCov@50 claim.
8. Move dynamic retrieval, old/negative experiments, and exploratory probes to
   appendix or brief limitations unless they improve the headline fixed-K
   results.
9. Render the PDF and check the visual page limit before each submission-ready
   handoff.

## Acceptance-Oriented Positioning

The best paper is not "our system solves everything." The best paper is:

> We identify the correct retrieval condition for exact verification, build a
> training objective aligned to that condition, show significant matched-budget
> gains on a hard graph retrieval benchmark, and clearly separate learned
> retrieval from exact solving.

That story is much harder to reject because it is technically precise, shows
real ablation evidence, and owns the limitations instead of fighting them.
