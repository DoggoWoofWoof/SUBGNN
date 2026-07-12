# Targeted Experiment Runbook (stop re-running the full grid)

The full production benchmark is **8 query families × 6 methods × 3 sizes × 2 seeds**.
Re-running all of it to iterate on *one* family (e.g. random_walk) or *one* operator
(selective / bridge overlap) wastes a full cloud billing hour. Use the cheapest tool that
answers the question.

## Tier 0 — offline probe ($0, no cloud, no solver, no encoder, seconds-to-minutes)

For anything about **overlap recall/size** (does the candidate contain the match? how big is
it?), you do NOT need the solver, the encoder, or raw features — only the cached overlap index
+ label tokens + a query set. Use `scripts/probe_selective_overlap.py`.

```bash
# random_walk only, simulate realistic retrieval (keep 75% of true partitions), all overlap policies
python scripts/probe_selective_overlap.py --mode incomplete --family random_walk \
  --keep-frac 0.75 --out runs/diagnostics/random_walk_overlap_fix_probe.csv

# size-at-fixed-recall (complete retrieval + noise budget)
python scripts/probe_selective_overlap.py --mode complete --budget-parts 50
```

Policies swept include `blunt`, `topk8`, `label`, and the random-walk fix `bridge2`,
`bridge2_only`, `bridge3`. Add `--family <type>` to restrict to one family.

## Tier 1 — targeted real run (one family, one method, no solver)

When you need the *real* learned ranking (neural) but only for one family, restrict the grid.
Add `--skip-solver` to measure retrieval recall / candidate size without the (cheap) solver,
or keep the solver for solve rate. This is ~1/48 of the full grid.

```bash
python scripts/benchmark_overlap_glasgow_cascade.py \
  --dataset mag --query-types random_walk --method hybrid \
  --target-sizes 20,50,100 --budgets 20,50,100,200,500,1000 \
  --overlap-bridge-infill-min 2 \
  --output-prefix runs/probes/mag_rw_bridge2 --cache-dir <mag_cache_dir> \
  --model mag_rgcn_best=<model_path> --skip-solver
```

Compare against the blunt baseline by dropping `--overlap-bridge-infill-min 2`.

## Tier 2 — full grid (only for the final paper table / once per locked config)

Only run all families × methods × seeds when producing the canonical summary, not while
iterating. The currently-running v5 job (corrected connected multi_fine/multi_coarse MAG) is a
legitimate Tier-2 run because those rows are still provisional in the paper.

## Selective / bridge overlap flags (benchmark_overlap_glasgow_cascade.py)

| Flag | Effect | Use for |
| --- | --- | --- |
| `--overlap-max-parts N` | per partition, keep top-N neighbors by support | shrink candidate (latency) |
| `--overlap-min-support N` | drop neighbors contributing < N boundary nodes | shrink candidate |
| `--overlap-max-nodes N` | global cap on overlap nodes | hard size control |
| `--overlap-label-compatible` | only add label-matching overlap nodes | cheap lossless shrink |
| `--overlap-bridge-infill-min N` | add full nodes of partitions bordering >= N selected parts | **random_walk / dispersed recall** |
| `--no-boundary-overlap` | drop 1-hop boundary overlap | bridge-infill-only experiments |

Empty policy == the original blunt one-hop union (safe drop-in; existing runs unaffected).
