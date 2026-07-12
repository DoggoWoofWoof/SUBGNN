# Jigsaw — System Architecture & Usage

A practical map of the codebase: what each module/script is, how to run the main
workflows, where artifacts live, and an honest note on the monolithic scripts and how
they should be decomposed. For the research narrative see the papers in `paper/`; for
number provenance see `benchmarks/paper_results/CANONICAL_SOURCES.md`.

---

## 1. The pipeline (what the system does)

Jigsaw is **retrieval-constrained exact subgraph matching**. One query flows through four stages:

```
          ┌─ retrieve ─┐   ┌─ stitch ──┐   ┌─ prune ───┐   ┌─ verify ──┐
 query ─▶ │ rank coarse│ ▶ │ +one-hop  │ ▶ │ typed sig │ ▶ │  Glasgow  │ ▶ match / no-match
          │ partitions │   │  overlap  │   │ + labels  │   │  (exact)  │
          └────────────┘   └───────────┘   └───────────┘   └───────────┘
             src/model      benchmark_*      benchmark_*     src/glasgow_solver
             (learned) +     cascade          cascade         + solver_registry
             classical
```

- **Retrieve** — a learned GNN encoder (or a classical label index / FAISS) ranks the
  graph's coarse partitions for the query. Coverage is the failure mode: if a required
  partition is missed, the answer is lost.
- **Stitch** — expand selected partitions by one-hop boundary overlap (recovers
  boundary-adjacent misses).
- **Prune** — drop candidate nodes by typed signatures + exact labels. Provably
  coverage-lossless (never removes a required node).
- **Verify** — run the Glasgow exact solver on the bounded candidate. Sound inside the
  candidate; the only failure mode is retrieval coverage, never a wrong match.

---

## 2. `src/` — the library (9 modules)

Import as `from src.<module> import ...` (scripts add the repo root to `sys.path`).

| module | role |
|---|---|
| `model.py` | Learned retriever encoders: `ImprovedSubgraphEncoder` (GIN, homogeneous graphs) and `RelationAwareSubgraphEncoder` (6-layer RGCN, heterogeneous MAG); `get_graph_embedding(graph, encoder, device)` returns `(graph_emb, node_emb)`. |
| `data.py` | Dataset loading (`load_dataset` for cora/arxiv/mag/…) and hierarchical partitioning (`build_single_hierarchy`, `make_partitions`, `make_undirected_with_edge_type`). Produces `coarse_graphs`, `fine_graphs`, `node_to_coarse_map`, `fine_to_coarse_map`. |
| `query_generator.py` | NetworkX-based **evaluation-time** query generation (interpretable). |
| `glasgow_solver.py` | Wrapper around the Glasgow Subgraph Solver — the exact verifier. |
| `subgraph_matching_solver.py` | Wrapper around the external SubgraphMatching binary (CFL/DP-iso/GQL) used for the resident-solver baselines. |
| `solver_registry.py` | Registry/dispatch over the available solvers (`glasgow`, SubgraphMatching variants). |
| `config.py` | Shared constants / config. |
| `utils.py` | Shared helpers used across the pipeline. |
| `test_solvers.py` | Manual solver smoke-test: `python -m src.test_solvers` builds a tiny known graph/query and checks each solver. Dev utility, not part of the pipeline. |

> Removed (dead code, recoverable from git history): `src/evaluate.py` (legacy Modal-only
> CLI eval, superseded by the `scripts/benchmark_*` cascade) and `src/sampling.py`
> (unimported training sampler; trainers now sample inline).

---

## 3. `scripts/` — grouped by role

Top-level `scripts/*.py` are the pipeline + tooling. `modal_*.py` / `patch_*.py` are
**legacy Modal** (gitignored, abandoned for Lightning) — do not use.

### Core pipeline / benchmarks
| script | what it does |
|---|---|
| `benchmark_overlap_glasgow_cascade.py` | **Main cascade** entry point: retrieve → overlap → prune → Glasgow, with per-family metrics. `--dataset --method --budgets …`. Contains FeatureIndex, mean-feature, selective/bridge overlap. *(monolith — §6)* |
| `benchmark_glasgow.py` | Glasgow exact-matching utilities + query generation + fine-embedding cache. *(monolith)* |
| `benchmark_retrieval.py` | Retrieval-only strategies: global fine embeddings, partition graphs, neighbor-stitch, multiview. *(monolith)* |
| `retrieval_strategies.py` | Pure ranking policies (no I/O): `reciprocal_rank_fusion`, `fine_parent_ranking`, `ranked_neighbor_stitch`, `hybrid_boundary_expand`. |
| `benchmark_non_neural.py` | Classical retrieval baselines (random, mean-feature, topo-feature, pagerank). |
| `coverage_losses.py` | **The FullCov objective** — `partition_coverage_loss` (CVaR-over-weakest-positives + top-K barrier). Tested by `tests/test_coverage_losses.py`. *(this is the loss to port for coverage-aware retrieval.)* |

### Training
`train_final_loss_local.py` (current local trainer, FullCov/CVaR), `lightning_rgcn_mag_train.py`
(MAG RGCN on Lightning), `train_jigsaw_model.py` *(monolith; older path)*.

### Research probes (inference-only, reusable)
`probe_fi_selectivity.py` (label-selectivity crossover / selector), `probe_finegrain_expansion.py`
(fine-granularity + overlap remedies), `probe_multivector_ranking.py` (subquery MaxSim),
`probe_selective_overlap.py`. Each writes a per-query CSV under `runs/`.

### Figures (canonical)
`generate_paper_figures_v2.py`, `generate_scaling_figure.py`, `generate_ablation_pareto_figures.py`.
⚠️ **Do not use `generate_submission_figures.py`** — it reads a stale run and is superseded.

### Summaries / analysis
`summarize_production_benchmarks.py` (per-query → summary), `analyze_candidate_shrinkage.py`,
`analyze_significance.py`, `compute_boundary_overlap_stats.py`, and
`analysis/reproduce_paper_numbers.py` (headline-number verifier).

### Lightning infrastructure
`lightning_production_benchmark.py` (job launcher: `prepare-package` / `launch-job`),
`lightning_cli_windows.py` (CLI shim), `lightning_mag_benchmark.py`,
`run_mag_benchmark_matrix_local.py`, `streaming_serve_smoke.py` (bounded-memory serve).
`launchers/` holds `.sh` runtime wrappers + `.ps1` job recipes.

---

## 4. How to run the main workflows

```bash
# Reproduce / verify the paper headline numbers (fast, local CSVs)
python scripts/analysis/reproduce_paper_numbers.py

# Run the exact-matching cascade on a dataset
python scripts/benchmark_overlap_glasgow_cascade.py \
    --dataset cora --method hybrid --budgets 2,5,10,20 --queries 50

# A research probe (inference-only; no solver): label-selectivity crossover
python scripts/probe_fi_selectivity.py --dataset cora --data-root data \
    --model models/cora-6_layer-model-graphsage_..._best_fullcov.pth --queries 15 \
    --output runs/fi_selectivity/cora_fi.csv

# Aggregate per-query results into a summary
python scripts/summarize_production_benchmarks.py "runs/<run>/results/*_per_query.csv" \
    --output runs/<run>/summary.csv

# Regenerate canonical figures
python scripts/generate_paper_figures_v2.py

# Launch a Lightning job (packages repo + overlay, runs a launcher script remotely)
python scripts/lightning_production_benchmark.py launch-job --owner <user> \
    --package-model <pkg> --code-patch-model <overlay> --run-script scripts/launchers/<x>.sh
```

---

## 5. Data & artifacts (what's committed vs local)

| location | contents | git |
|---|---|---|
| `benchmarks/paper_results/` | **Canonical evidence** — summary CSVs, `CANONICAL_SOURCES.md`, `final_results/HEADLINE_NUMBERS.csv`, manifests | **committed** (force-included) |
| `paper/` | manuscripts (`.tex`, `.pdf`), figures, tables | committed |
| `models/` | checkpoints (`*.pth`); deployed MAG = `mag-...-rgcn_seed7203_walkaware_best_fullcov_DEPLOYED.pth` | **gitignored** (`*.pth`) except `README.md` |
| `data/` | datasets (Cora local; Arxiv/MAG auto-download via OGB) | gitignored |
| `runs/` | per-query outputs, probe CSVs, run logs | gitignored (local provenance) |
| `cache/` | regenerable hierarchy / embedding caches | gitignored |

**Convention:** `runs/` is local working data; anything a paper number depends on is copied
into `benchmarks/paper_results/` and listed in `CANONICAL_SOURCES.md`.

---

## 6. Reproducibility

Every paper number maps to exactly one source:
- `benchmarks/paper_results/CANONICAL_SOURCES.md` — number → script + CSV, single source of truth.
- `final_results/HEADLINE_NUMBERS.csv` — tidy mirror of the production matrix + selector + foreclosure.
- `scripts/analysis/reproduce_paper_numbers.py` — runnable checker (MAG matrix, selector, foreclosure).
- `docs/log2026_submission_checklist.md` — submission SHA, page/anonymity checks.

---

## 7. Known tech debt: the monolith scripts (and how to split them)

Four committed scripts are "god scripts" (>2k lines each) that accreted as the project grew:

| script | lines | should become |
|---|---|---|
| `train_jigsaw_model.py` | 3048 | `train/` package: data-sampling · loss (import `coverage_losses`) · loop · checkpointing |
| `benchmark_glasgow.py` | 2289 | `matching/`: query-gen · Glasgow drive · fine-embedding cache · metrics |
| `benchmark_overlap_glasgow_cascade.py` | 2272 | `cascade/`: hierarchy load · retrieval (feature/mean/learned) · overlap · pruning · solve-drive · metrics |
| `benchmark_retrieval.py` | 2054 | `retrieval/`: index build · strategies (import `retrieval_strategies`) · fine embeddings · eval |

**Why not refactored here:** these scripts generate the **frozen, reproducible paper
numbers**. A safe decomposition must (a) preserve every entry point / CLI flag, and
(b) re-pass `scripts/analysis/reproduce_paper_numbers.py` and the LoG SHA check afterward.
That is a deliberate, separately-verified task — not a drive-by edit. Until then, they are
documented above so the codebase is navigable despite the monoliths. The pure-logic pieces
have *already* been extracted and are the right templates for the rest:
`retrieval_strategies.py` (ranking policies) and `coverage_losses.py` (the objective).
