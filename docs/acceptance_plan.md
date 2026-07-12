# Jigsaw — Path to a Clear Accept (deadlines + readiness)

## STATUS (2026-06-22)
- DONE: #3a MAG connected rerun merged (neural 64.6%->86.0%; multi_coarse 2.3%->71.7%); #4
  statistical rigor (95% CIs + paired McNemar, `tab:significance`, neural sig. > all baselines
  incl Mean-RRF p=0.014); #5 hetero dataset (MAG is heterogeneous/RGCN — corrected); #6 novelty
  framing; #3b random_walk bound-and-explained + walk-aware retrain staged; claim audit closed;
  canonical artifact consistent; repo reorganized.
- COMPUTE-GATED (staged, need cloud): #2 Pareto run (selective overlap end-to-end on MAG —
  build-time win pre-validated locally at 2.5x); #1 external baseline (classical full-graph
  Glasgow ready via flags; learned NeuroMatch/GLSearch is effort); more training seeds.
- The two compute-gated items are the only remaining must-haves for a clear A* accept.

---


Date: 2026-06-21. Maps each acceptance requirement to feasibility, what it needs,
the exact command, and effort. Status legend: READY (runnable now with existing
code), CLOUD (needs a Lightning/GCP run; config provided), EFFORT (new code/training),
STRETCH (major new work).

## Upcoming venues (target order)

| Venue | Deadline | Conf date | Fit | Notes |
|---|---|---|---|---|
| **PVLDB / VLDB 2027** | **Rolling, 1st of each month** (Vol 20, through Mar 1 2027) | Aug 2027, Athens | **Best** (graph data management, indexing for exact search) | Revise-and-resubmit model — most forgiving; submit a fall month (Sep 1 / Oct 1 2026) after the two must-haves land. **Primary target.** |
| **WSDM 2027** | **Aug 24, 2026** (~9 wks) | Feb 2027 | Good (search/retrieval + data mining) | Selective, short. Only if the baseline + Pareto sprint finishes in time. **Stretch-primary.** |
| **ICDE 2027** | **Nov 11, 2026** | May 2027, Copenhagen | Strong (DB) | Comfortable runway; good fallback/parallel DB target. |
| **KDD 2027 (Cycle 1)** | **Feb 1, 2027** | Aug 2027, San Jose | Good (data mining) | Spacious; two-cycle, R&R-friendly. |
| CIKM 2026 | May 23, 2026 — **PASSED** | — | Good | Missed; CIKM 2027 ~May 2027. |

**Recommendation:** anchor on **PVLDB rolling** (submit Sep/Oct 2026), keep **ICDE 2027 (Nov 11)**
as the parallel DB option, and treat **WSDM 2027 (Aug 24)** as a sprint target only if the two
must-haves are done by early August. SIGIR/ICML are poor fits — do not target.

## Must-have #1 — external baseline (#1 reject risk)

- **Classical control: full-graph exact Glasgow with a time budget — READY.** No code change:
  `--method all --no-overlap --signature none` triggers the full-graph fast path (solver runs on
  the entire graph, timeout = `--solver-timeout`). This is the "exact matching without
  retrieval/partitioning" baseline; on MAG it will mostly time out, which is precisely the
  evidence that retrieval is necessary. Run it per dataset/family/size alongside the matrix.
  ```bash
  python scripts/benchmark_overlap_glasgow_cascade.py --dataset mag \
    --query-types single,k_hop,random_walk,degree_k_hop --target-sizes 20,50,100 \
    --method all --no-overlap --signature none --solver-timeout 30 \
    --output-prefix runs/baselines/mag_direct_glasgow --cache-dir <mag_cache>
  ```
- **Learned external (NeuroMatch / GLSearch): EFFORT.** Node-alignment systems; adapt their
  node scores into a partition ranking, or run them on candidate regions. Strongest reviewer
  signal but heaviest. Do this only if targeting WSDM/KDD where a learned competitor is expected;
  for VLDB the classical control + the internal policy sweep is usually sufficient.

## Must-have #2 — a clear Pareto win (the experiment that most changes reception)

- **Selective overlap end-to-end on MAG — CLOUD, config ready.** Flags are wired
  (`--overlap-max-parts`, `--overlap-label-compatible`). Run neural with selective overlap vs
  blunt and FilterAll; plot (positive solve rate) vs (total latency) and (candidate nodes).
  Offline probe already shows selective overlap cuts the overlap union 3–6×, so candidate-build
  time (the dominant cost, 3.4 s vs 0.25 s solve) should drop enough for **total latency to beat
  FilterAll at comparable recall** — that is the Pareto win to demonstrate.
  ```bash
  # selective (Jigsaw+): neural ranking + per-partition top-8 neighbors + label-compatible overlap
  python scripts/benchmark_overlap_glasgow_cascade.py --dataset mag --method hybrid \
    --query-types all --target-sizes 20,50,100 --budgets 20,50,100,200,500,1000 \
    --overlap-max-parts 8 --overlap-label-compatible --prune-query-labels --component-solve \
    --signature type_rel_feat32 --solver-timeout 5 \
    --model mag_rgcn_best=<model_path> --cache-dir <mag_cache> \
    --output-prefix runs/pareto/mag_neural_selective
  # blunt baseline: same minus the two overlap flags  -> runs/pareto/mag_neural_blunt
  ```
  Then `scripts/generate_submission_figures.py` gains a Pareto panel (recall vs latency, recall vs
  candidate nodes) — add it once both runs land.
- **Pre-check (READY, $0):** confirm the latency hypothesis locally before the cloud run by timing
  candidate construction under blunt vs selective on the cached index (no solver/encoder needed) —
  extend `probe_selective_overlap.py` to record `candidate_build_seconds`.

## Must-have #3 — resolve the two visible failures

- **multi_coarse 2.3% — CLOUD, in progress.** The running v5 connected rerun replaces the broken
  disconnected rows. On completion: download to `runs/lcr_mag_v3` (short path), validate 300
  unique queries/method with `query_component_count==1`, run `summarize_production_benchmarks.py`,
  and merge `{mag} × {multi_fine, multi_coarse}` into the canonical summary + paper family table.
- **random_walk 70% — bound-and-explain (DONE) + retrieval-side fix (EFFORT).** The overlap-side
  bridge-infill fix was tested and **falsified** ($0 probe: +1pp only) — the bottleneck is
  retrieval ranking, not overlap. Paper now states this honestly. The real fix is retrieval-side:
  (a) budget scaled by query span, (b) walk-aware training positives/hard-negatives — requires
  retraining the MAG encoder (EFFORT). For the next submission, bound-and-explain is defensible;
  the retrain is a strong-acceptance lift.

## Strong-acceptance lifts

- **#4 More seeds + CIs — CLOUD, easy.** Add seeds beyond 20260607/08 (e.g. 20260609/10) to the
  grid and report 95% CIs / paired tests on the production matrix (extend the FullCov McNemar
  rigor already in the paper).
- **#5 Non-homogenized scale/diversity — STRETCH.** Reviewers will say homogenized MAG inflates
  difficulty. Add a heterogeneous graph where relations matter (DBLP-HetG is the recommended
  option per `runs/diagnostics/r1_r2_findings.md`); needs a new loader in `src/data.py`.
- **#6 Novelty framing — DONE.** Paper now leads with the two reviewer-resistant results:
  pruning is provably coverage-lossless (all recall loss is at the overlap stage) and the overlap
  operator is tight per-partition (5.4×) while only the union over the budget is broad (the 921K
  correction). Add the Pareto result as the third headline once #2 lands.

## Sequencing for the next ~8 weeks (to hit a fall VLDB month / WSDM)

1. (now) Let v5 finish → merge corrected MAG multi rows (#3a). 
2. (week 1) Local candidate-build timing pre-check (#2 pre-check). 
3. (week 1–2) Launch the two Pareto runs (selective vs blunt) + DirectGlasgow baseline (#1, #2). 
4. (week 2–3) Add Pareto figure + baseline rows to the paper; extend seeds (#4). 
5. (week 3–4) Decide WSDM-Aug-24 sprint vs VLDB-fall: if Pareto win is clean, sprint WSDM; else
   target a VLDB fall month and add the learned external baseline (#1) and/or DBLP-HetG (#5).
