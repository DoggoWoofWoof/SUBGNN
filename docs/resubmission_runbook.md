# Resubmission Runbook

This runbook starts from the cleaned repository state and targets the main reviewer concerns: benchmark clarity, coverage validation, stronger ablations, and more careful exactness claims.

For the complete mathematical definition of the final loss, worked examples,
and the fixed/dynamic retrieval pipeline, see
[`jigsaw_loss_and_retrieval_method.md`](jigsaw_loss_and_retrieval_method.md).
For the June 9 ECML PKDD / MLG manuscript strategy, including the main-paper
claim boundaries, table plan, and reviewer-concern mapping, see
[`ecml_pkdd_mlg_paper_strategy.md`](ecml_pkdd_mlg_paper_strategy.md).
The same-model connected-query-view retrieval ablation and its negative
primary-method conclusion are documented in
[`multiview_retrieval_ablation.md`](multiview_retrieval_ablation.md).

## 1. Current Evidence

Canonical current paper results live in:

- `benchmarks/paper_results/glasgow_benchmark_corafull_all.csv`
- `benchmarks/paper_results/glasgow_benchmark_arxiv_all.csv`
- `benchmarks/paper_results/manifest.json`

Legacy diagnostics and old benchmark generations are archived under `archive/legacy_benchmarks/`.

## 2. Best Arxiv Coverage Retrain

The current best completed Arxiv model is:

- run name: `coverage_v2_allpos_fresh`
- model: `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v2_allpos_fresh.pth`
- checkpoint: `/cache/arxiv_coverage_v2_allpos_fresh_checkpoint.pth`
- local training log: `runs/logs/coverage_v2_allpos_fresh_direct_20260606_011834.out.log`

This run biases training toward K-hop and multi-coarse cases, keeps all true
K-hop coarse partitions as coverage targets, and bounds only the positive
context graph for memory.

```powershell
python modal_experiment_pipeline.py --mode best-arxiv --fresh
```

Default important settings:

- `gamma_partition=1.0`
- `prob_k_hop=0.45`
- `prob_multi_coarse=0.35`
- `prob_single_part=0.10`
- `max_train_coarse_parts=50`
- `max_gpos_nodes=4000`
- benchmark `top_k=20,50,100`

Expected output model:

- `models/arxiv-6_layer-model-jigsaw_coverage_v1.pth`

Expected Modal result files:

- `results/glasgow_benchmark_arxiv_all_k20_coverage_v1.csv`
- `results/glasgow_benchmark_arxiv_all_k50_coverage_v1.csv`
- `results/glasgow_benchmark_arxiv_all_k100_coverage_v1.csv`

### Fine-Coverage V3 Fine-Tune

This run starts from the best v2 checkpoint and adds a fine-partition coverage
loss. It should be compared against v2, not automatically adopted.

```powershell
.\.venv_modal\Scripts\modal.exe run --detach scripts/train_jigsaw_model.py --dataset arxiv --epochs 40 --batch-size 8 --steps-per-epoch 75 --num-hierarchies 1 --run-name coverage_v3_finecov_from_v2_e40 --gamma-partition 1.5 --gamma-fine-partition 0.5 --coverage-temperature 0.05 --fine-cache-refresh-steps 250 --alpha 0.2 --beta 0.0 --prob-k-hop 0.6 --prob-single-part 0.05 --prob-multi-coarse 0.25 --max-gpos-nodes 4000 --max-train-coarse-parts 80 --cache-refresh-steps 50 --learning-rate 0.00002 --scheduler-type plateau --min-learning-rate 0.000005 --warmup-steps 0 --plateau-patience 5 --plateau-factor 0.5 --resume-from-checkpoint /cache/arxiv_coverage_v2_allpos_fresh_checkpoint.pth --resume-model-only --fresh
```

Completed v3 run metadata:

- completed app id: `ap-wwDdgkeGtdFqSRqxETRT0B`
- completed function call: `fc-01KTDQGFFFC5T9S0NRCX9VH9D3`
- local wrapper log: `runs/logs/coverage_v3_finecov_from_v2_e40_resume_20260606_111641.out.log`
- local training log: `runs/logs/train_arxiv_coverage_v3_finecov_from_v2_e40.log`
- volume training log: `/cache/logs/train_arxiv_coverage_v3_finecov_from_v2_e40.log`
- checkpoint: `/cache/arxiv_coverage_v3_finecov_from_v2_e40_checkpoint.pth`
- final model: `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v3_finecov_from_v2_e40.pth`
- final epoch: Avg Loss `10.174313`, CoarsePart `4.299868`, FinePart `6.581912`, LR `5.0e-06`

Historical Modal logs:

```powershell
.\.venv_modal\Scripts\modal.exe app logs ap-wwDdgkeGtdFqSRqxETRT0B --function-call fc-01KTDQGFFFC5T9S0NRCX9VH9D3 --tail 200 --timestamps
```

## 3. K Sweep Only

Use this after a model already exists locally.

```powershell
python modal_experiment_pipeline.py --mode topk-sweep --dataset arxiv --bench-only --run-name coverage_v1 --top-ks 20,50,100
```

For Cora, use K values before the full-graph fallback effect dominates:

```powershell
python modal_experiment_pipeline.py --mode topk-sweep --dataset cora --bench-only --run-name default --top-ks 1,5,10,20 --queries 100
```

### Arxiv K-Hop Fine-Boundary Benchmark

Use this section for the urgent Arxiv k-hop table. The preferred diagnostic now
starts from coarse seed `K=20` and expands dynamically; the older `K=125/150`
rows remain useful as an upper diagnostic, but should not be the headline
method.

#### Preferred top-20 seed dynamic retrieval

This keeps neural FAISS retrieval fixed at `K=20`, then expands from those
seeds over the coarse boundary graph and verifies fine-boundary candidates.

Template command:

```powershell
.\.venv_modal\Scripts\modal.exe run --detach modal_benchmark_glasgow.py --dataset arxiv --queries 30 --query-type k_hop --target-sizes 20 --top-k 20 --faiss-score-k 20 --boundary-expand-coarse-budget 100 --solver-timeout 45 --model-path /cache/models/arxiv-6_layer-model-jigsaw_coverage_v2_allpos_fresh.pth --output-tag coverage_v2_allpos_fresh_k20seed_fbe2_b100_prune_q30_seed42 --seed 42 --include-oracle --skip-full --stitch-strategy fine_boundary_expand --stitch-levels 25,50,75,100,150,200,250,300,400,625,750 --stitch-seed-count 20 --require-candidate-fullcov --prune-target-by-query-labels
```

For unpruned verification, omit `--prune-target-by-query-labels`.

Completed corrected v2 q30 top-20 seed results:

| Method | Seed K | Expansion budget | Label prune | FullCov@SeedK | Expanded FullCov | Candidate FullCov | Stitch solved | Oracle solved | Avg recall@SeedK | Timeouts |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fine boundary expand | 20 | 50 | no | 9/30 | 14/30 | 14/30 | 14/30 | 30/30 | 0.6847 | 0 |
| fine boundary expand | 20 | 75 | no | 9/30 | 17/30 | 17/30 | 16/30 | 30/30 | 0.6847 | 1 |
| fine boundary expand | 20 | 100 | no | 9/30 | 20/30 | 20/30 | 16/30 | 30/30 | 0.6847 | 4 |
| fine boundary expand | 20 | 50 | yes | 9/30 | 14/30 | 14/30 | 14/30 | 30/30 | 0.6847 | 0 |
| fine boundary expand | 20 | 75 | yes | 9/30 | 17/30 | 17/30 | 17/30 | 30/30 | 0.6847 | 0 |
| fine boundary expand | 20 | 100 | yes | 9/30 | 20/30 | 20/30 | 20/30 | 30/30 | 0.6847 | 0 |

Local corrected CSVs:

- `runs/logs/glasgow_benchmark_arxiv_k_hop_k20_coverage_v2_allpos_fresh_k20seed_fbe2_b50_unpruned_q30_seed42.csv`
- `runs/logs/glasgow_benchmark_arxiv_k_hop_k20_coverage_v2_allpos_fresh_k20seed_fbe2_b75_unpruned_q30_seed42.csv`
- `runs/logs/glasgow_benchmark_arxiv_k_hop_k20_coverage_v2_allpos_fresh_k20seed_fbe2_b100_unpruned_q30_seed42.csv`
- `runs/logs/glasgow_benchmark_arxiv_k_hop_k20_coverage_v2_allpos_fresh_k20seed_fbe2_b50_prune_q30_seed42.csv`
- `runs/logs/glasgow_benchmark_arxiv_k_hop_k20_coverage_v2_allpos_fresh_k20seed_fbe2_b75_prune_q30_seed42.csv`
- `runs/logs/glasgow_benchmark_arxiv_k_hop_k20_coverage_v2_allpos_fresh_k20seed_fbe2_b100_prune_q30_seed42.csv`

Interpretation:

- Dynamic expansion from top-20 is useful: b100 improves candidate FullCov from
  `9/30` to `20/30`.
- It is not enough for a final strong claim: `10/30` queries still miss at
  least one required partition after expansion.
- Label pruning is verifier-side only. It shows that exact verification is fast
  after candidate coverage is achieved, but should not be described as neural
  retrieval.
- Unpruned b100 has the same `20/30` candidate FullCov but only `16/30` solved
  because Glasgow times out on 4 candidates.

#### High-K diagnostic rows

These rows use much deeper initial retrieval and label-pruned exact
verification. They are useful diagnostics but should not be the main dynamic
retrieval claim.

```powershell
.\.venv_modal\Scripts\modal.exe run --detach modal_benchmark_glasgow.py --dataset arxiv --queries 30 --query-type k_hop --target-sizes 20 --top-k 125 --solver-timeout 45 --model-path /cache/models/arxiv-6_layer-model-jigsaw_coverage_v2_allpos_fresh.pth --output-tag coverage_v2_allpos_fresh_fine_boundary_k125_prune_fixed_q30_seed42 --seed 42 --include-oracle --skip-full --stitch-strategy fine_boundary --stitch-levels 25,50,75,100,150,200,250,300,400,625 --stitch-seed-count 10 --require-candidate-fullcov --prune-target-by-query-labels
```

```powershell
.\.venv_modal\Scripts\modal.exe run --detach modal_benchmark_glasgow.py --dataset arxiv --queries 30 --query-type k_hop --target-sizes 20 --top-k 150 --solver-timeout 45 --model-path /cache/models/arxiv-6_layer-model-jigsaw_coverage_v2_allpos_fresh.pth --output-tag coverage_v2_allpos_fresh_fine_boundary_k150_prune_fixed_q30_seed42 --seed 42 --include-oracle --skip-full --stitch-strategy fine_boundary --stitch-levels 25,50,75,100,150,200,250,300,400,625,750 --stitch-seed-count 10 --require-candidate-fullcov --prune-target-by-query-labels
```

Completed v2 q30 results:

- K=125: stitch `26/30`, FullCov `26/30`, oracle `30/30`, avg recall `0.9703`, zero timeouts.
- K=150: stitch `28/30`, FullCov `28/30`, oracle `30/30`, avg recall `0.9860`, zero timeouts.

Completed v3 q30 ablation results:

- K=125: stitch `24/30`, FullCov `24/30`, oracle `30/30`, avg recall `0.9664`, zero timeouts.
- K=150: stitch `26/30`, FullCov `26/30`, oracle `30/30`, avg recall `0.9743`, zero timeouts.
- Decision: do not adopt v3 as the paper model; it is weaker than v2 on FullCov@K and stitch success.

Local v3 benchmark artifacts:

- `runs/logs/glasgow_benchmark_arxiv_k_hop_k125_coverage_v3_finecov_from_v2_e40_fine_boundary_k125_prune_q30_seed42.csv`
- `runs/logs/glasgow_benchmark_arxiv_k_hop_k150_coverage_v3_finecov_from_v2_e40_fine_boundary_k150_prune_q30_seed42.csv`
- `runs/logs/benchmark_arxiv_k_hop_k125_coverage_v3_finecov_from_v2_e40_fine_boundary_k125_prune_q30_seed42.log`
- `runs/logs/benchmark_arxiv_k_hop_k150_coverage_v3_finecov_from_v2_e40_fine_boundary_k150_prune_q30_seed42.log`

Modal v3 benchmark artifacts:

- `gnn-data-volume:/data/results/glasgow_benchmark_arxiv_k_hop_k125_coverage_v3_finecov_from_v2_e40_fine_boundary_k125_prune_q30_seed42.csv`
- `gnn-data-volume:/data/results/glasgow_benchmark_arxiv_k_hop_k150_coverage_v3_finecov_from_v2_e40_fine_boundary_k150_prune_q30_seed42.csv`
- `gnn-data-volume:/data/logs/benchmark_arxiv_k_hop_k125_coverage_v3_finecov_from_v2_e40_fine_boundary_k125_prune_q30_seed42.log`
- `gnn-data-volume:/data/logs/benchmark_arxiv_k_hop_k150_coverage_v3_finecov_from_v2_e40_fine_boundary_k150_prune_q30_seed42.log`

Compare completed CSVs:

```powershell
python scripts/compare_benchmark_csvs.py --run v2_k125=runs\logs\glasgow_benchmark_arxiv_k_hop_k125_coverage_v2_allpos_fresh_fine_boundary_k125_prune_fixed_q30_seed42.csv --run v2_k150=runs\logs\glasgow_benchmark_arxiv_k_hop_k150_coverage_v2_allpos_fresh_fine_boundary_k150_prune_fixed_q30_seed42.csv --run v3_k125=runs\logs\glasgow_benchmark_arxiv_k_hop_k125_coverage_v3_finecov_from_v2_e40_fine_boundary_k125_prune_q30_seed42.csv --run v3_k150=runs\logs\glasgow_benchmark_arxiv_k_hop_k150_coverage_v3_finecov_from_v2_e40_fine_boundary_k150_prune_q30_seed42.csv
```

### Node-Alignment V4 Continuation

This run attacks the current bottleneck: weak FullCov@20 seed retrieval. It
starts from v2, disables the failed fine-partition coverage loss, and enables
the existing node-alignment loss with `beta=0.10`.

```powershell
.\.venv_modal\Scripts\modal.exe run --detach scripts\train_jigsaw_model.py --dataset arxiv --epochs 40 --batch-size 8 --steps-per-epoch 75 --num-hierarchies 1 --run-name coverage_v4_nodebeta_from_v2_e40 --gamma-partition 1.5 --gamma-fine-partition 0.0 --coverage-temperature 0.05 --alpha 0.15 --beta 0.10 --prob-k-hop 0.70 --prob-single-part 0.03 --prob-multi-coarse 0.22 --max-gpos-nodes 4000 --max-train-coarse-parts 100 --cache-refresh-steps 50 --learning-rate 0.00002 --scheduler-type plateau --min-learning-rate 0.000005 --warmup-steps 0 --plateau-patience 5 --plateau-factor 0.5 --resume-from-checkpoint /cache/arxiv_coverage_v2_allpos_fresh_checkpoint.pth --resume-model-only --fresh
```

Active run metadata:

- active app id: `ap-l8o8oBY4S2fJN7QKFzqbYV`
- checkpoint: `/cache/arxiv_coverage_v4_nodebeta_from_v2_e40_checkpoint.pth`
- final model: `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v4_nodebeta_from_v2_e40.pth`
- volume log: `/cache/logs/train_arxiv_coverage_v4_nodebeta_from_v2_e40.log`

Early status:

- no OOM/traceback during launch and first epoch
- epoch 1 Avg Loss `7.363656`
- epoch 1 CoarsePart `4.403702`
- GPU memory `0.25/1.55 GB`
- coverage targets observed around avg/max `8.8/19` and `9.0/21`

Decision rule: adopt v4 only if it improves `FullCov@SeedK` and
`Expanded/Candidate FullCov` against the corrected v2 top-20 seed table. If v4
does not improve those, keep v2 as the paper model and present v4 as a negative
ablation.

### Top-K Barrier V5 Continuation

This run is the direct model-side fix for weak `FullCov@20`. The previous
all-positive coverage loss pushes every true partition upward, but it does not
explicitly enforce that all positives sit above enough negatives to fit inside
top-20. V5 adds a differentiable FullCov@K surrogate:

- if a query touches at most 20 true coarse partitions, optimize for top-20
- if it touches 21-30 partitions, optimize for top-30
- if it touches 31-40 partitions, optimize for top-40
- continue in buckets of 10 for broader k-hop examples

This keeps broad k-hop rows in training instead of dropping them, while avoiding
an impossible top-20 constraint when the query itself needs more than 20
partitions.

Launch command:

```powershell
.\.venv_modal\Scripts\modal.exe run --detach scripts\train_jigsaw_model.py --dataset arxiv --epochs 40 --batch-size 8 --steps-per-epoch 75 --num-hierarchies 1 --run-name coverage_v5_topkbarrier_from_v2_e40 --gamma-partition 1.5 --gamma-fine-partition 0.0 --coverage-temperature 0.05 --coverage-topk 20 --coverage-topk-weight 0.35 --coverage-topk-margin 0.0 --alpha 0.15 --beta 0.05 --prob-k-hop 0.70 --prob-single-part 0.03 --prob-multi-coarse 0.22 --max-gpos-nodes 4000 --max-train-coarse-parts 100 --cache-refresh-steps 50 --learning-rate 0.00002 --scheduler-type plateau --min-learning-rate 0.000005 --warmup-steps 0 --plateau-patience 5 --plateau-factor 0.5 --resume-from-checkpoint /cache/arxiv_coverage_v2_allpos_fresh_checkpoint.pth --resume-model-only --fresh
```

Active Modal state:

- run name: `coverage_v5_topkbarrier_from_v2_e40`
- active app id: `ap-EIVIZ5xvoObQihlgdZyCAi`
- source checkpoint: `/cache/arxiv_coverage_v2_allpos_fresh_checkpoint.pth`
- checkpoint: `/cache/arxiv_coverage_v5_topkbarrier_from_v2_e40_checkpoint.pth`
- expected final model: `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v5_topkbarrier_from_v2_e40.pth`
- expected volume log: `/cache/logs/train_arxiv_coverage_v5_topkbarrier_from_v2_e40.log`

Confirmed startup config:

- `coverage_topk=20`
- `coverage_topk_weight=0.35`
- `coverage_topk_margin=0.0`
- `resume_model_only=True`
- optimizer and scheduler reset for this fine-tune

Early status:

- no OOM/traceback during startup
- epoch 1 Avg Loss `7.753064`, CoarsePart `4.739801`, GPU memory `0.26/1.13 GB`
- epoch 2 Avg Loss `7.737020`, CoarsePart `4.729276`, GPU memory `0.26/1.59 GB`

Decision rule: adopt v5 only if it improves seed `FullCov@20` and downstream
expanded/candidate FullCov against both v2 and v4. If it only improves loss but
not fixed-seed FullCov, it is a negative ablation.

## 4. Coverage Ablation

Run this to prove whether the partition coverage loss is doing real work.

```powershell
python modal_experiment_pipeline.py --mode coverage-ablation --dataset cora --epochs 40 --queries 25 --top-ks 5,10,20 --gamma-values 0,0.1,0.5,1.0 --fresh
```

If budget allows, repeat on Arxiv with fewer queries first:

```powershell
python modal_experiment_pipeline.py --mode coverage-ablation --dataset arxiv --epochs 60 --queries 25 --top-ks 20,50 --gamma-values 0,0.5,1.0 --fresh --skip-full
```

## 5. Additional Dataset Probe

This addresses reviewer concern about only Cora/Arxiv. Start with small query counts.

```powershell
python modal_experiment_pipeline.py --mode datasets --datasets pubmed,physics,citeseer --epochs 60 --queries 25 --top-ks 20,50 --fresh
```

### MAG Follow-Up After Arxiv Is Final

Do not start MAG until the Arxiv control/scheduler comparison is resolved. MAG is now wired through the same training and benchmark path, using a type-aware homogeneous representation:

- paper OGB features
- node-type one-hot features
- per-relation in/out degree features
- feature schema: `mag_type_rel_v1`
- hierarchy cache: `/cache/mag_hierarchies_type_rel_v1.pt`

Recommended first MAG smoke run:

```powershell
python modal_experiment_pipeline.py --mode datasets --datasets mag --run-name mag_type_rel_smoke --epochs 20 --steps-per-epoch 40 --batch-size 8 --queries 10 --top-ks 50 --target-sizes 20,50 --query-type k_hop --gamma-partition 2.0 --prob-k-hop 0.70 --prob-multi-coarse 0.25 --prob-single-part 0.02 --max-gpos-nodes 2500 --max-train-coarse-parts 50 --cache-refresh-steps 10 --learning-rate 0.0002 --scheduler-type cosine --min-learning-rate 0.00005 --warmup-steps 0 --cosine-t-max 20 --fresh --skip-full
```

If the smoke run is stable, run a longer MAG probe:

```powershell
python modal_experiment_pipeline.py --mode datasets --datasets mag --run-name mag_type_rel_v1 --epochs 120 --steps-per-epoch 75 --batch-size 8 --queries 25 --top-ks 50,100 --target-sizes 20,50,100 --gamma-partition 2.0 --prob-k-hop 0.70 --prob-multi-coarse 0.25 --prob-single-part 0.02 --max-gpos-nodes 2500 --max-train-coarse-parts 50 --cache-refresh-steps 10 --learning-rate 0.0002 --scheduler-type cosine --min-learning-rate 0.00005 --warmup-steps 0 --cosine-t-max 120 --fresh --skip-full
```

## Final V2/V4/V5 Retrieval-Only Decision

The final fixed-seed comparison was run without Glasgow. It evaluates the same
30 Arxiv k-hop queries for every model and measures retrieval directly:

- fixed neural retrieval at `K=20,50,100`
- top-20 neural seeds followed by boundary expansion to budgets `50,75,100`
- coarse FullCov and recall
- fine-pool and fine-boundary FullCov

Local evidence:

- `runs/logs/retrieval_arxiv_khop_v2_v4_v5_q30_seed42_summary.csv`
- `runs/logs/retrieval_arxiv_khop_v2_v4_v5_q30_seed42_per_query.csv`
- `runs/logs/retrieval_arxiv_khop_v2_v4_v5_q30_seed42.remote.log`

| Model | FullCov@20 | FullCov@50 | FullCov@100 | Dynamic FullCov B50 | B75 | B100 | Avg Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 all-positive coverage | 9/30 | 13/30 | **23/30** | **14/30** | 17/30 | 20/30 | 0.6847 |
| V4 node-beta continuation | 9/30 | 15/30 | 20/30 | **14/30** | **18/30** | **21/30** | **0.7013** |
| V5 top-K barrier | 9/30 | **16/30** | 20/30 | 13/30 | 17/30 | **21/30** | 0.6879 |

Fine-boundary FullCov at fine budget 100 after coarse expansion budget 100:

- V2: `13/30`
- V4: `10/30`
- V5: `10/30`

Decision:

- Keep **V2** as the main paper model. It remains best at fixed `K=100` and
  retains the strongest fine-level coverage.
- Report V4 and V5 as mixed/negative ablations. They improve some mid-budget
  coarse FullCov rows, but neither improves seed `FullCov@20`.
- Dynamic boundary expansion is useful: the best result grows from `9/30`
  FullCov at the top-20 seed to `21/30` at budget 100.
- Do not claim the new loss solved retrieval. The primary bottleneck remains
  neural seed ranking because all models stay at `9/30 FullCov@20`.
- Do not ensemble V4 and V5 for dynamic budget 100. They recover the same one
  additional query beyond V2; their union is still `21/30`.
- Glasgow is not required for selecting the retrieval model. Keep only a small
  final end-to-end verification table to demonstrate that fully covered
  candidates can be verified.

## Hybrid Retrieval And V6 Screening

The next experiments target the two remaining causes of failed FullCov:

1. The old boundary expansion followed the local frontier and used neural rank
   only as a tie-breaker. It could discard useful partitions ranked 21-100.
2. The mean all-positive loss diluted the single worst required partition,
   although that partition determines FullCov failure.

Implemented retrieval-only ablation:

- seed with top-20 coarse neural retrieval
- score frontier candidates with both neural rank and boundary support
- periodically teleport to the best remaining neural candidate
- retrieve fine partitions globally, map them to coarse parents, and fuse that
  ranking with the coarse ranking using reciprocal-rank fusion
- sweep model weights `0.25,0.5,0.75`, teleport intervals `5,10`, and final
  coarse budgets `50,75,100`
- do not invoke Glasgow

Launch:

```powershell
.\scripts\run_hybrid_retrieval_modal.ps1
```

Implemented V6 screening objective:

- CVaR aggregation over the worst 25% of required positive partitions
- dynamic top-K barrier remains `20,30,40,...` according to positive count
- up to 24 true coarse partitions are re-encoded live per batch so coverage
  gradients reach the positive partition encoder instead of only stale cache
- fixed 30-query k-hop validation every two epochs
- best checkpoint selected lexicographically by FullCov@20, FullCov@50,
  Recall@20, then Recall@50

Launch:

```powershell
.\scripts\run_v6_screen_modal.ps1
```

Adopt V6 only if its fixed validation and held-out retrieval-only benchmark
improve FullCov@20 beyond `9/30`. A lower training loss alone is not evidence.

### Completed V2 hybrid/global-fine retrieval result

The retrieval-only sweep completed on the same 30 fixed-seed Arxiv k-hop
queries without invoking Glasgow.

| Method | Seed FullCov@20 | FullCov B50 | FullCov B75 | FullCov B100 | Avg recall B100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old top-20 boundary expansion | 9/30 | 14/30 | 17/30 | 20/30 | 0.9248 |
| Fixed neural top-100 | - | - | - | 23/30 | 0.9429 |
| Coarse hybrid, model weight 0.5, teleport 10 | 9/30 | 14/30 | 20/30 | **25/30** | 0.9601 |
| Global-fine RRF hybrid, model weight 0.5, teleport 10 | **10/30** | 15/30 | **21/30** | **25/30** | **0.9626** |

Important interpretation:

- Hybrid expansion improves the old dynamic B100 result from `20/30` to
  `25/30`, and also beats blind fixed top-100 retrieval (`23/30`).
- Global fine-parent fusion improves the initial top-20 FullCov from `9/30` to
  `10/30` and gives the strongest average B100 recall.
- The best individual B100 method misses five queries. The union of all tested
  hybrid methods covers `26/30`, so one additional query is recoverable by a
  better deterministic fusion policy.
- Use the global-fine RRF hybrid with model weight `0.5` and teleport interval
  `10` as the current default retrieval method. Keep the coarse-only hybrid as
  an ablation showing that most of the gain comes from combining neural rank
  with boundary support.

Evidence:

- `runs/logs/retrieval_arxiv_khop_hybrid_globalfine_v2_q30_seed42_summary.csv`
- `runs/logs/retrieval_arxiv_khop_hybrid_globalfine_v2_q30_seed42_per_query.csv`

### Completed V6 screening run

V6 completed 20 epochs without OOM, traceback, cancellation, or checkpoint
failure. It fine-tuned the V2 encoder with the CVaR/live-positive objective and
reset optimizer and scheduler state.

| Check | Result |
| --- | ---: |
| Lowest training loss | `6.2172` at epoch 10 |
| Lowest coarse-partition loss | `5.2981` at epoch 10 |
| Best validation FullCov@20 | `9/30` at epoch 10 |
| Best validation FullCov@50 | `15/30` at epoch 6 |
| Best validation Recall@20 | `0.6941` at epoch 16 |
| Best validation Recall@50 | `0.8473` at epoch 18 |
| Peak GPU memory | `2.22 GB` |

The loss decreased, but V6 did not produce a clear fixed-validation FullCov@20
gain during training. The held-out seed-42 retrieval-only comparison did,
however, show a consistent improvement over V2:

| Model | Fixed FullCov@20 | Fixed FullCov@50 | Fixed FullCov@100 | Avg recall@100 |
| --- | ---: | ---: | ---: | ---: |
| V2 | 9/30 | 13/30 | 23/30 | 0.9429 |
| V6 best-validation checkpoint | 10/30 | 15/30 | 25/30 | 0.9673 |
| **V6 final checkpoint** | **11/30** | **16/30** | **26/30** | **0.9761** |

V6-final also gives the strongest single dynamic-selection curve:

| Selector | Seed FullCov@20 | FullCov B50 | FullCov B75 | FullCov B100 | Avg recall B100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 global-fine hybrid | 10/30 | 15/30 | 21/30 | 25/30 | 0.9626 |
| **V6-final coarse hybrid, weight 0.5, teleport 10** | **11/30** | **17/30** | **20/30** | **26/30** | **0.9719** |

The larger 100-query, new-seed confirmation preserves the model-ranking gain
but exposes an important selector tradeoff:

| Model/method | FullCov@20 | FullCov@50 | FullCov@75 | FullCov@100 |
| --- | ---: | ---: | ---: | ---: |
| V2 fixed neural ranking | 24/100 | 48/100 | - | 81/100 |
| V6-final fixed neural ranking | **25/100** | **53/100** | - | **85/100** |
| V2 best dynamic method | 25/100 | 53/100 | 71/100 | **89/100** |
| V6-final best dynamic method | **27/100** | **60/100** | **75/100** | 87/100 |

V6-final is therefore the better learned ranker and the better constrained-
budget retriever through B75. V2 still expands more effectively at B100 as a
single model. Do not claim V6 dominates every retrieval configuration. Their
complementary failures motivated the completed deterministic cross-model RRF
selector below.

### Final two-seed retrieval recommendation

The deterministic cross-model selector was evaluated on a second independent
100-query seed after locking the method
`cross_model_coarse_rrf_hybrid_mw0.75_teleport10`. The first seed's exploratory
best of `92/100` did not repeat exactly, but the locked method still beat the
same fixed V2 hybrid comparator on both seeds at B100: `92 vs 89`, then
`84 vs 83`.

Aggregate fixed neural retrieval across the two 100-query seeds:

| Model | FullCov@20 | FullCov@50 | FullCov@100 | Recall@20 | Recall@50 | Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 | 49/200 | 95/200 | 155/200 | 0.6932 | 0.8614 | 0.9611 |
| **V6-final** | **53/200** | **106/200** | **163/200** | **0.7151** | **0.8862** | **0.9660** |

Aggregate dynamic retrieval using fixed named methods:

| Method | FullCov B50 | FullCov B75 | FullCov B100 | Recall B100 |
| --- | ---: | ---: | ---: | ---: |
| V2 coarse hybrid, weight 0.5, teleport 10 | 102/200 | 133/200 | 172/200 | 0.9740 |
| V6 global-fine hybrid, weight 0.75, teleport 5 | 110/200 | 137/200 | 166/200 | 0.9701 |
| **V2+V6 cross-model coarse RRF hybrid, weight 0.75, teleport 10** | **109/200** | **141/200** | **176/200** | **0.9781** |

Final recommendation:

- use V6-final as the primary learned model
- use cross-model coarse RRF hybrid as the strongest dynamic retrieval result
  when the extra cost of running V2 and V6 is acceptable
- include V2 coarse hybrid as the single-model efficiency baseline
- report all fixed methods and fixed named dynamic methods across both seeds;
  do not report per-seed oracle unions or per-budget cherry-picked selectors as
  the headline

### V7 continuation and clean reproduction

Two final-objective training runs are used for different scientific purposes:

1. `coverage_v7_continue_from_v6` tests whether a low-learning-rate
   continuation can improve the strongest existing model.
2. `coverage_v7_clean_seed7002` starts from random initialization and tests
   whether the finalized V6 objective is independently reproducible.

Both runs use:

- the finalized V6 CVaR/live-positive/top-K-barrier objective
- `100` training steps per epoch and batch size `8`
- a fixed validation suite of `100` k-hop queries: 50 from seed `31415` and 50
  from seed `27182`
- checkpoint ordering by FullCov@20, FullCov@50, FullCov@100, Recall@20,
  Recall@50, then Recall@100
- validation every two epochs and early stopping after four non-improving
  validation checks
- explicit training seeds for reproducibility

Initial validation confirms that the runs are distinct:

| Run | Initialization | FullCov@20 | FullCov@50 | FullCov@100 |
| --- | --- | ---: | ---: | ---: |
| V7 continuation | V6-final | 22/100 | 46/100 | 79/100 |
| V7 clean | random, seed 7002 | 1/100 | 8/100 | 19/100 |

Adoption rules:

- replace V6-final only if the V7 continuation best checkpoint improves
  held-out retrieval across the fixed two-seed evaluation suite
- use the clean run as the primary reproducibility model only if it approaches
  the continuation/V6 result without checkpoint inheritance
- if the clean run is weaker, report staged V2-to-V6 training honestly and use
  the clean run as a reproducibility limitation or ablation
- do not tune parameters against the final independent retrieval seeds

Current screening outcome:

| Run | Status | Baseline FullCov@20/50/100 | Best validation FullCov@20/50/100 | Interpretation |
| --- | --- | ---: | ---: | --- |
| V7 clean seed 7002 | Early-stopped at epoch 12 | 1/8/19 | 16/40/70 | Confirms the objective learns from scratch, but this shortened run is neither the adopted model nor a fair causal comparison against V2's approximately 9,000 steps. |
| V7 continuation from V6 | Completed epoch 16 | 22/46/79 | 26/46/77 | Improves the priority FullCov@20 key; evaluate best and final checkpoints on locked held-out retrieval before adoption. |

V7 clean artifacts:

- best validation model:
  `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_clean_seed7002_best_fullcov.pth`
- final model:
  `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_clean_seed7002.pth`
- checkpoint: `/cache/arxiv_coverage_v7_clean_seed7002_checkpoint.pth`
- log: `/cache/logs/train_arxiv_coverage_v7_clean_seed7002.log`

V7 continuation artifacts:

- best validation model:
  `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth`
- final model:
  `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6.pth`
- checkpoint: `/cache/arxiv_coverage_v7_continue_from_v6_checkpoint.pth`
- log: `/cache/logs/train_arxiv_coverage_v7_continue_from_v6.log`

The locked held-out retrieval launcher is
`scripts/run_v7_retrieval_modal.ps1`. It evaluates V6-final and all V7
best/final checkpoints on independent 100-query seeds `20260607` and
`20260608`. Launching it requires explicit approval to upload the current
private code and consume the private model artifacts in the external
`pilgnnteam` Modal workspace.

Locked held-out retrieval outcome across both 100-query seeds:

| Model | Fixed FullCov@20 | Fixed FullCov@50 | Fixed FullCov@100 | Recall@20 | Recall@50 | Recall@100 | Avg max true rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V6-final | 53/200 | 106/200 | 163/200 | 0.7151 | 0.8862 | 0.9660 | 57.42 |
| **V7 continuation best** | **55/200** | **106/200** | **168/200** | **0.7170** | **0.8866** | **0.9690** | **55.03** |
| V7 continuation final | 55/200 | 105/200 | 168/200 | 0.7163 | 0.8886 | 0.9687 | 55.35 |
| V7 clean best | 34/200 | 78/200 | 163/200 | 0.5820 | 0.8263 | 0.9607 | 66.20 |
| V7 clean final | 41/200 | 86/200 | 162/200 | 0.6285 | 0.8462 | 0.9604 | 63.28 |

Locked single-model dynamic selector
`coarse_hybrid_mw0.5_teleport10`:

| Model | FullCov B50 | FullCov B75 | FullCov B100 | Recall B100 |
| --- | ---: | ---: | ---: | ---: |
| V6-final | 106/200 | 142/200 | 167/200 | 0.9653 |
| **V7 continuation best** | **108/200** | **142/200** | **168/200** | 0.9651 |
| V7 continuation final | 108/200 | 139/200 | 166/200 | 0.9639 |

Adopt `V7 continuation best` as the staged primary checkpoint. Its gain over
V6-final is modest and is not uniform on every individual seed, so describe it
as an aggregate held-out improvement rather than universal dominance. This
does not establish that the objective is causally better than V2; that claim
requires the matched 9,000-step runs.

The two fixed query sets touched an average of 6.47 and 6.27 true coarse
partitions, with maxima 17 and 16. Thus no query was mathematically impossible
at K=20, 50, or 100. Current retrieval CSVs record model-load and FAISS-build
time, but not per-query retrieval latency; add explicit timing before producing
the final paper latency table.

Launcher:

```powershell
.\scripts\run_v7_training_modal.ps1 -Mode continuation
.\scripts\run_v7_training_modal.ps1 -Mode clean
```

Artifacts:

- best model:
  `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v6_cvar_livepos_from_v2_best_fullcov.pth`
- final model:
  `/cache/models/arxiv-6_layer-model-jigsaw_coverage_v6_cvar_livepos_from_v2.pth`
- checkpoint: `/cache/arxiv_coverage_v6_cvar_livepos_from_v2_checkpoint.pth`
- log: `/cache/logs/train_arxiv_coverage_v6_cvar_livepos_from_v2.log`
- held-out launcher: `scripts/run_v6_retrieval_modal.ps1`
- held-out q30 summary:
  `runs/logs/retrieval_arxiv_khop_hybrid_consensus_v2_v6_q30_seed42_summary.csv`
- held-out q30 per-query:
  `runs/logs/retrieval_arxiv_khop_hybrid_consensus_v2_v6_q30_seed42_per_query.csv`
- new-seed q100 summary:
  `runs/logs/retrieval_arxiv_khop_hybrid_consensus_v2_v6_q100_seed20260607_summary.csv`
- new-seed q100 per-query:
  `runs/logs/retrieval_arxiv_khop_hybrid_consensus_v2_v6_q100_seed20260607_per_query.csv`
- cross-model q100 seed 20260607 summary:
  `runs/logs/retrieval_arxiv_khop_crossmodel_v2_v6_q100_seed20260607_summary.csv`
- cross-model q100 seed 20260608 summary:
  `runs/logs/retrieval_arxiv_khop_crossmodel_v2_v6_q100_seed20260608_summary.csv`

### Final experimental protocol and fairness

The existing V2 checkpoint trained for 120 epochs at approximately 75 steps
per epoch, or about 9,000 optimizer steps. Therefore, the current 4,000-step
V7 clean run is a screening and reproducibility probe, not a fair causal
comparison against V2. Early stopping is fair only when every compared
objective receives the same maximum optimizer-step budget and uses the same
validation suite, checkpoint ordering, and patience.

Keep the following claims separate:

1. **Staged system improvement:** compare the existing V2 checkpoint, V6-final,
   and the V7 continuation. This measures whether the complete development
   sequence improved retrieval, but it does not isolate the effect of the V6
   objective because later runs inherit earlier weights.
2. **Matched objective ablation:** train the control and final objectives from
   random initialization for the same maximum 9,000 optimizer steps. Use the
   same initialization seeds, query generator, batch size, sample mixture,
   learning-rate schedule, validation queries, and checkpoint selector. Run
   every objective for the full matched budget; only the named objective
   component may change.
3. **Retrieval-system ablation:** evaluate fixed neural retrieval and dynamic
   selectors using the same frozen model checkpoint and fixed query sets. This
   isolates retrieval logic from training changes.

#### Locked final model definition

The matched-budget result locks the following from-scratch objective and
architecture for the paper and all new datasets:

- 6-layer residual GIN graph encoder
- 256 hidden dimensions, dropout `0.1`, and layer normalization
- per-layer mean, max, and sum pooling
- normalized 128-dimensional graph embeddings
- `0.2 * fine_NCE + 0.8 * coarse_NCE`
- all-positive coarse coverage loss with CVaR aggregation
- `coverage_temperature=0.05`
- `coverage_cvar_fraction=0.25`
- bucketed FullCov barrier beginning at `coverage_topk=20`
- `coverage_topk_weight=0.35`, `coverage_topk_margin=0.0`
- `max_live_positive_parts=24`
- `gamma_partition=1.0`, `gamma_fine_partition=0.0`

The bucketed barrier uses K=20 when a query touches at most 20 true
partitions, K=30 for 21-30 partitions, K=40 for 31-40 partitions, and so on.
It never drops a broad training query merely because FullCov@20 is impossible.
For other datasets, the initial K is scaled to approximately 10% of the number
of coarse partitions and the bucket increment is scaled to approximately 5%.
Arxiv retains its evaluated `coverage_topk=20` and
`coverage_topk_bucket_size=10` behavior exactly.

#### Matched-budget objective ablations

Use exactly 9,000 optimizer steps for every from-scratch objective run and
disable early stopping for this comparison. Use at least two identical
training seeds across variants if compute permits. Report both the best
validation-selected checkpoint and the final-step checkpoint. The minimal
main-paper ablation is:

| Variant | Positive aggregation | Top-K barrier weight | Live positives |
| --- | --- | ---: | ---: |
| Matched control | mean | 0.0 | 0 |
| + worst-positive emphasis | CVaR, fraction 0.25 | 0.0 | 0 |
| + FullCov barrier | CVaR, fraction 0.25 | 0.35 | 0 |
| Final objective | CVaR, fraction 0.25 | 0.35 | 24 |

Freeze architecture, partition hierarchy, query generation, training mixture,
batch size, optimizer, learning-rate schedule, and total optimizer-step budget.
If compute remains, ablate `coverage_topk_weight` over `{0.2, 0.35, 0.5}` and
`coverage_cvar_fraction` over `{0.1, 0.25, 0.5}`. Do not make fine-partition
loss, node-level loss, MC dropout, or MCTS-style retrieval part of the primary
method unless a fixed-seed retrieval experiment first demonstrates a clear
gain; current evidence does not justify them.

Matched-run launcher examples:

```powershell
.\scripts\run_final_objective_ablation_modal.ps1 -Variant control -TrainingSeed 7101
.\scripts\run_final_objective_ablation_modal.ps1 -Variant cvar -TrainingSeed 7101
.\scripts\run_final_objective_ablation_modal.ps1 -Variant topk -TrainingSeed 7101
.\scripts\run_final_objective_ablation_modal.ps1 -Variant final -TrainingSeed 7101
```

Repeat the same four variants with seed `7102` for the second replicate. Each
launcher invocation runs exactly 90 epochs by 100 steps with early stopping
disabled.

#### Matched-budget execution recovery (June 7, 2026)

The first six matched-budget jobs in `pilgnnteam` ended remotely without an
OOM or Python traceback at approximately epochs 48--60. Subsequent app
creation was rejected because that workspace reached its billing-cycle spend
limit. The six committed checkpoints and the exact cached Arxiv partition
hierarchy were preserved and migrated to the explicitly approved
`deepalimohapatra1973` workspace.

The resumed jobs load the original `/cache/arxiv_hierarchies_finecov_v1.pt`
before loading their matching model, optimizer, scheduler, and global-step
state. Verified resume points are:

| Variant | Seed | Resume epoch | Global step |
| --- | ---: | ---: | ---: |
| control | 7101 | 56 | 5600 |
| control | 7102 | 54 | 5400 |
| cvar | 7101 | 54 | 5400 |
| topk | 7101 | 58 | 5800 |
| final | 7101 | 46 | 4600 |
| final | 7102 | 48 | 4800 |

A migration smoke attempt initially detected that the fallback workspace did
not yet have the original hierarchy and began constructing a new one. That app
was stopped before it completed an epoch or committed a checkpoint. The
original hierarchy was then uploaded and every active resume was verified to
load it. Consequently, the continued runs retain the original partitioning and
matched-budget protocol.

#### Locked matched-budget result

All six runs completed exactly 9,000 optimizer steps. Both locked independent
query seeds (`20260607` and `20260608`) were then evaluated with 100 queries
each. There are no impossible-at-K rows: the queries touch an average of 6.37
true coarse partitions, with a maximum of 17.

The primary causal comparison uses each run's validation-selected checkpoint.
Control and final-objective results pool two training seeds each; CVaR-only and
top-K-only are single-seed component probes.

| Objective | Fixed FullCov@20 | Fixed FullCov@50 | Fixed FullCov@100 | Avg max true rank |
| --- | ---: | ---: | ---: | ---: |
| control, mean positives | 87/400 (21.8%) | 156/400 (39.0%) | 264/400 (66.0%) | 74.02 |
| CVaR only | 38/200 (19.0%) | 85/200 (42.5%) | 132/200 (66.0%) | 74.52 |
| CVaR + top-K barrier | 33/200 (16.5%) | 71/200 (35.5%) | 128/200 (64.0%) | 81.53 |
| **final: CVaR + barrier + live positives** | **104/400 (26.0%)** | **195/400 (48.8%)** | **328/400 (82.0%)** | **58.97** |

The complete objective improves fixed FullCov over the matched control by
`+4.25`, `+9.75`, and `+16.0` percentage points at K=`20/50/100`. Both final
training replicates outperform both control replicates at every fixed K. Fixed
recall also improves from `0.6620/0.8319/0.9384` to
`0.7012/0.8740/0.9676`.

Because every checkpoint is evaluated on the same queries, a paired exact
McNemar comparison is also available. At fixed K=`20/50/100`, final uniquely
covers `25/58/86` queries that the matched control misses, while control
uniquely covers only `8/19/22`; two-sided exact p-values are `0.0046`,
`9.8e-06`, and `4.0e-10`. Thus the fixed-ranking gain is not merely an
unpaired aggregate-count fluctuation.

The locked single-model dynamic selector
`coarse_hybrid_mw0.5_teleport10`, seeded at K=20, produces:

| Objective | FullCov B=50 | FullCov B=75 | FullCov B=100 |
| --- | ---: | ---: | ---: |
| control | 171/400 (42.8%) | 248/400 (62.0%) | 315/400 (78.8%) |
| **final** | **187/400 (46.8%)** | **264/400 (66.0%)** | **321/400 (80.2%)** |

The dynamic selector adds useful early-budget coverage, but its advantage
narrows at B=100 because both methods approach the same broad candidate
region. The model-quality headline should therefore remain fixed neural
ranking, with dynamic retrieval reported as the efficient system result.
Paired dynamic wins/losses at B=`50/75/100` are `40/24`, `52/36`, and
`32/26`; their exact p-values (`0.060`, `0.109`, and `0.512`) do not establish
a statistically clear dynamic-selector improvement.

The component probes do **not** support claiming that CVaR or the top-K barrier
alone improves retrieval. Their benefit appears only when combined with live
positive partition re-encoding in the complete objective. This is consistent
with the intended mechanism: the worst-positive and rank-boundary losses need
fresh gradients through the true partition embeddings.

For the clean from-scratch model, adopt the validation-selected checkpoint:

`/cache/models/arxiv-6_layer-model-jigsaw_coverage_final_ablation_final_seed7101_best_fullcov.pth`

It was selected before held-out evaluation and obtains fixed FullCov
`51/96/165` and locked dynamic FullCov `95/133/160` over the 200 held-out
queries. The second training replicate is reported for robustness.

Keep the staged operational checkpoint separate:

`/cache/models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth`

The staged V7 continuation remains the strongest currently measured
single-model system (`55/106/168` fixed and `108/142/168` locked dynamic), but
because it is checkpoint continuation rather than a matched from-scratch run,
it is **not** causal evidence for the final objective. Use the clean final
objective for all future from-scratch dataset training.

The locked CSVs do not contain per-query retrieval latency, so no latency claim
should be made from this evaluation. The reproducible aggregate is generated
by:

```powershell
python scripts/analyze_final_ablation_retrieval.py
```

Execution priority when compute or submission time is limited:

1. **Required:** matched `control` and `final` runs for seeds `7101` and
   `7102`.
2. **Required component evidence:** `cvar` and `topk` for seed `7101`.
3. **Preferred robustness:** repeat `cvar` and `topk` for seed `7102`.
4. **Optional:** top-K weight and CVaR-fraction sweeps.

#### Cross-dataset training protocol

After the Arxiv objective is locked, train one final single-model checkpoint
from scratch on each additional dataset. Start with a short smoke run, then use
the same matched-step and validation-selection protocol. Keep the objective
fixed. For MAG, report the strict Arxiv operating points first, then append
partition-count-normalized retrieval diagnostics.

| Dataset | Coarse partitions | Fine partitions per coarse | Initial role |
| --- | ---: | ---: | --- |
| Cora | 20 | 5 | Small-graph full benchmark and sanity check |
| CiteSeer | 10 | 5 | Small additional generalization result |
| PubMed | 20 | 5 | Medium citation-network result |
| Physics | 35 | 5 | Larger homogeneous-network result |
| MAG | 2000 | 5 | Heterogeneous-to-homogeneous scalability result |

The reusable launcher fixes the adopted objective and scales its initial
FullCov barrier:

| Dataset | Initial coverage K | Bucket increment | Full-run command |
| --- | ---: | ---: | --- |
| Cora | 2 | 1 | `.\scripts\run_cross_dataset_final_modal.ps1 -Dataset cora -TrainingSeed 7201` |
| CiteSeer | 1 | 1 | `.\scripts\run_cross_dataset_final_modal.ps1 -Dataset citeseer -TrainingSeed 7201` |
| PubMed | 2 | 1 | `.\scripts\run_cross_dataset_final_modal.ps1 -Dataset pubmed -TrainingSeed 7201` |
| Physics | 4 | 2 | `.\scripts\run_cross_dataset_final_modal.ps1 -Dataset physics -TrainingSeed 7201` |
| MAG | 20 | 10 | `.\scripts\run_cross_dataset_final_modal.ps1 -Dataset mag -TrainingSeed 7201` |

Run the corresponding `-Smoke` command first in a new workspace. Full runs use
90 epochs by 100 steps, the same final objective, and no early stopping. The
default training query-size schedule is biased toward practical 20-node queries
without becoming 20-only: `20,20,20,50,100` with jitter `5`. Override this with
`-QueryTargetSizes` and `-QuerySizeJitter` only for an explicit ablation, e.g.
`-QueryTargetSizes "20" -QuerySizeJitter 0`. MAG automatically uses the
implemented `mag_type_rel_v1` homogeneous feature schema. Its hierarchy cache is
`/cache/mag_hierarchies_type_rel_2000_fine5_finecov_v1.pt`.

Use normalized retrieval budgets for cross-dataset comparability:

- fixed K at approximately 10%, 25%, and 50% of the coarse partitions
- dynamic retrieval seeded at approximately 10%
- dynamic budgets at approximately 25%, 37.5%, and 50%
- full-graph K only as a capacity/oracle diagnostic, never as the headline

For example, Arxiv uses fixed K=`20/50/100` and dynamic seed/B=`20 ->
50/75/100`. MAG first preserves the same strict Arxiv operating points
`20/50/100`, then appends scale-normalized diagnostics `200/500/1000` for its
2,000 coarse partitions. Cora uses fixed K=`2/5/10` and dynamic seed/B=`2 ->
5/8/10`. Also report how many queries touch more true partitions than each K,
because FullCov is mathematically impossible for those rows.

For MAG, use the implemented homogeneous representation with base/paper
features, node-type one-hot features, and per-relation in/out-degree features.
Name this feature schema explicitly (`mag_type_rel_v1`) and state that relation
types are encoded as features rather than preserved as typed message-passing
edges.

#### Final retrieval and benchmark matrix

Primary deployment/retrieval reporting:

- **Primary single model:** the adopted final checkpoint with fixed neural
  ranking and its locked single-model dynamic hybrid.
- **High-cost optional result:** locked V2+final cross-model coarse RRF hybrid;
  report its doubled model/index cost and do not present it as the default.
- **Primary metric:** FullCov@K. Report recall@K as a secondary diagnostic.
- **Fixed retrieval:** K=20, 50, and 100 on Arxiv.
- **Dynamic retrieval:** seed K=20 with locked budgets B=50, 75, and 100.
- **Diagnostics:** average/max true partition count, maximum true-partition
  rank, missed-partition count, and candidate size. Add retrieval latency only
  after timing instrumentation is present in the result CSVs.
- **Query evaluation:** fixed query files or fixed seeds, with multiple
  independent seeds. Never choose a different selector per seed or budget.
- **Verification:** run Glasgow only on a small end-to-end table after
  retrieval FullCov analysis. Describe it as exact within the retrieved
  candidate region.

The final paper tables should contain:

1. matched-budget objective ablation on Arxiv
2. staged V2-to-final system comparison, clearly labeled
3. fixed and dynamic retrieval comparison on the same held-out queries
4. at least Cora plus one larger additional dataset, preferably MAG
5. a small end-to-end verification table and an oracle diagnostic separated
   from the retrieval headline

Final adoption is now locked: use the validation-selected clean
`final_seed7101_best_fullcov` checkpoint for causal objective reporting and all
future from-scratch dataset recipes. Use V7 continuation best only as the
separately labeled strongest staged operational checkpoint.

#### ECML PKDD June 9 benchmark addendum

The paper should use the new results to answer the reviewer concerns directly:
benchmark clarity, stronger ablations, coverage-aware evaluation, exactness
scope, and dataset breadth. The old Cora/Arxiv Glasgow tables can remain, but
they should no longer be the only evidence. The main-paper benchmark package
should add the following tables in priority order.

| Priority | Benchmark to add | Purpose | Status for submission |
| --- | --- | --- | --- |
| 1 | Matched-budget Arxiv objective ablation | Shows the final objective is better than the matched control under equal 9,000-step training budget. | Paper-ready from locked seeds `20260607/20260608`. |
| 1 | Arxiv fixed retrieval FullCov@20/50/100 | Establishes model retrieval quality without Glasgow confounds. | Paper-ready; FullCov primary, recall secondary. |
| 1 | Retrieval-selector baselines | Shows learned ranking beats non-neural candidate selection, which is the correct benchmark family beyond Glasgow. | Add random top-K, graph-neighbor expansion, and degree/PageRank/community heuristics where time permits. |
| 1 | Arxiv dynamic retrieval `20 -> 50/75/100` | Shows the practical system selector under a fixed, validation-locked budget policy. | Paper-ready; report as system result, not causal objective proof. |
| 1 | Query feasibility and difficulty | Prevents misleading recall-only claims by reporting true partition count, impossible-at-K, and max true-partition rank. | Paper-ready for Arxiv; add for every new dataset. |
| 2 | Query-type breakdown | Shows behavior on `single`, `multi_fine`, `multi_coarse`, and `k_hop`, instead of over-indexing on k-hop. | Required code path exists in Glasgow benchmark; retrieval-only benchmark still needs `--query-type all` support before final multi-type retrieval tables. |
| 2 | Small end-to-end Glasgow verification table | Demonstrates exact verification after successful retrieval while keeping model selection retrieval-only. | Use a small held-out subset only after candidate FullCov is known; do not run Glasgow for model selection. |
| 2 | Cross-dataset retrieval table | Addresses dataset-breadth concern beyond Cora/Arxiv. | Cora is available; MAG training completed but is currently a weak scalability diagnostic; PubMed/Physics/CiteSeer are fallback candidates if MAG retrieval remains too weak. |
| 3 | Multi-view/query-part reranking ablation | Documents that query-part occurrence/averaging was tested and did not robustly beat fixed retrieval. | Appendix/negative ablation only. |
| 3 | MC dropout/MCTS-style retrieval probes | Documents exploratory attempts without making them part of the method. | Appendix or omit unless a locked fixed-seed gain is produced. |

Main-paper Arxiv tables should use the clean from-scratch final objective,
`final_seed7101_best_fullcov`, for causal claims. The staged
`coverage_v7_continue_from_v6_best_fullcov` checkpoint may be reported as the
strongest operational checkpoint only in a separately labeled staged-system
comparison. Do not compare staged continuation against the matched control as
causal evidence.

Recommended main-paper table layout:

1. **Objective ablation table:** control, CVaR-only, CVaR+barrier, and final
   objective. Report fixed FullCov@20/50/100, recall@20/50/100, average max
   true-partition rank, and the paired McNemar result for final vs control.
2. **Retriever table:** fixed neural ranking and locked dynamic retrieval on
   the same Arxiv held-out queries. Report FullCov, recall, candidate budget,
   and candidate partition count. State that current CSVs do not yet include
   per-query retrieval latency, so latency must not be claimed from these
   rows.
3. **Retrieval-selector baseline table:** compare learned fixed top-K against
   random top-K, graph-neighbor expansion without learning, and cheap
   structural heuristics. This is the important benchmark family beyond
   Glasgow because it tests whether the learned retriever adds localization
   value before exact verification.
4. **Query-type stress table:** rows for `single`, `multi_fine`,
   `multi_coarse`, and `k_hop`, each at query target sizes `20/50/100`.
   Columns should be FullCov@20/50/100, recall@20/50/100, average true
   partitions, maximum true partitions, and impossible-at-K count.
5. **Exact verification table:** small, fixed subset with Glasgow run only
   after retrieval. Columns should be candidate FullCov, solved count,
   timeout count, stitched nodes, and solver time. Phrase exactness as
   "exact inside the retrieved candidate region".
6. **Cross-dataset table:** one row per dataset using the adopted final
   objective. Use fixed K=`20/50/100` everywhere for comparability, then add
   normalized budgets where appropriate. For MAG with 2,000 coarse partitions,
   report strict K=`20/50/100` first and normalized K=`200/500/1000` second.

Concrete Arxiv exact-verification evidence currently available:

| Setting | Candidate policy | Candidate FullCov | Solved | Failures after candidate FullCov | Median solver time | Paper use |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V2 coarse K=50 probe | 50 coarse partitions | 3/6 | 3/6 | 0 | 42.1s | Shows practical K=50 verification works when FullCov holds, but under-covers. |
| V2 coarse K=100 probe | 100 coarse partitions | 4/6 | 1/6 | 3 | 89.7s | Shows higher recall with large unpruned candidates can still fail the solver. |
| V2 fine-boundary K=125 q30 | FullCov-gated, label-pruned | 26/30 | 26/30 | 0 | 0.02s | Strong conditional Arxiv verification row. |
| V2 fine-boundary K=150 q30 | FullCov-gated, label-pruned | 28/30 | 28/30 | 0 | 0.02s | Strong conditional Arxiv verification row. |
| Final dynamic 20->50 q30 | Fine-boundary expansion, label-pruned | 11/30 | 11/30 | 0 | 0.00s | Low-budget final-model system point; still under-covers difficult K-hop queries. |
| Final dynamic 20->100 q30 | Fine-boundary expansion, label-pruned | 20/30 | 20/30 | 0 | 0.02s | Practical dynamic final-model system point; appendix if space is tight. |

This table should be interpreted as conditional exact verification, not as a
global Arxiv Glasgow speedup table. The full graph is too large for a stable
full-graph Glasgow baseline. The strongest Arxiv main-paper claim remains the
matched-budget fixed FullCov gain, especially K=50 (`39.0% -> 48.8%`, paired
exact `p=9.8e-06`). The final-model Arxiv Glasgow rerun completed on June 8,
2026 for `final_seed7101_best_fullcov`: seed K=20, B=50 and B=100, 30 held-out
queries, FullCov gating, label pruning, oracle rows, and no full-graph
Glasgow. B=50 solved/candidate-covered `11/30`; B=100 solved/candidate-covered
`20/30`; oracle solved `30/30`; there were zero solver failures after
candidate FullCov.

Ready-to-use headline evidence for the June 9 manuscript:

| Result | Control | Final objective | Defensible claim |
| --- | ---: | ---: | --- |
| Fixed FullCov@20 | 87/400 (21.8%) | 104/400 (26.0%) | `+4.25` percentage points; paired exact `p=0.0046`. |
| Fixed FullCov@50 | 156/400 (39.0%) | 195/400 (48.8%) | `+9.75` percentage points; paired exact `p=9.8e-06`. |
| Fixed FullCov@100 | 264/400 (66.0%) | 328/400 (82.0%) | `+16.0` percentage points; paired exact `p=4.0e-10`. |
| Dynamic FullCov B=50 | 171/400 (42.8%) | 187/400 (46.8%) | Useful system gain, but not statistically clear (`p=0.060`). |
| Dynamic FullCov B=75 | 248/400 (62.0%) | 264/400 (66.0%) | Useful system gain, but not statistically clear (`p=0.109`). |
| Dynamic FullCov B=100 | 315/400 (78.8%) | 321/400 (80.2%) | Similar broad-budget performance (`p=0.512`). |

This supports a precise result narrative: the complete objective significantly
improves the neural partition ranking, especially at larger fixed K. Dynamic
boundary expansion improves practical coverage from a small seed, but it
narrows the difference between objectives and should be reported as a system
component rather than evidence that every loss component independently helps.
The component probes show that CVaR-only and top-K-barrier-only do not improve
the matched control; the gain appears when they are combined with live-positive
partition re-encoding.

The June 9 rank-neighbor stitching probe should not replace this dynamic
retrieval story. It starts from a top-ranked seed set, admits only top-100
neighbor partitions in the coarse partition graph, and optionally uses
query/partition mean-feature similarity as a secondary rank. On
GraphSAGE+FullCov with locked seeds `20260607/20260608`, 100 queries per target
size, it only improves the 20-node K=20 bucket (`57/200` fixed to `62/200`
FullCov) and otherwise matches or trails the existing hybrid at the practical
K=50/100 budgets. Keep it as future work or an internal negative probe; do not
spend main-paper space on it. Aggregate artifacts are
`runs/logs/retrieval_arxiv_khop_stitch_v1_graphsage_q100_sizes20_50_100_aggregate.md`
and
`runs/logs/retrieval_arxiv_khop_stitch_v1_graphsage_q100_sizes20_50_100_aggregate_summary.csv`.

The follow-up prefix-seeded dynamic probe tested the more precise version of
the idea: use precision@1/2/5/10 to choose trusted anchor partitions, then
expand the bucket through neighbor-aware hybrid or stitching. This behaved as
expected mechanically: seed precision was high, especially for larger query
targets (`P@1` about `0.73/0.92/0.94` for target sizes `20/50/100`). However,
better anchors only produced small FullCov gains at the practical B=50 budget:
old dynamic to prefix-seeded hybrid was `106 -> 108` for 20-node queries,
`54 -> 55` for 50-node queries, and `24 -> 25` for 100-node queries, each out
of 200 locked queries. At B=100 the result ties or slightly trails the old
dynamic selector. Conclusion: precision-aware seeding is a promising future
retrieval knob, but it is not strong enough to add to the June 9 main paper.
Artifact:
`runs/logs/retrieval_arxiv_khop_prefix_seed_v1_graphsage_q100_sizes20_50_100_aggregate.md`.

For the June 9 deadline, do not wait for MAG before finalizing the central
paper tables. Lock the paper around:

1. the matched-budget Arxiv objective result above;
2. fixed Arxiv retrieval at K=`20/50/100`, with the K=50 gain treated as a
   confident practical-budget result;
3. Cora end-to-end exact verification and speedup;
4. the four-query-type stress analysis;
5. MAG only as an appendix scalability diagnostic if its result CSVs become
   available and are clearly labeled as weak/negative.

If an additional positive dataset result is still required, prioritize PubMed
or Physics because the current MAG objective has already shown weak validation
coverage and its retrieval evaluation is unavailable. A rushed MAG success
claim would be less defensible than a clearly reported negative scalability
result.

MAG should be handled carefully. The completed MAG training run did not show
useful validation coverage (`best FullCov@20/50/100 = 2/4/6 out of 100`, final
about `2/3/6 out of 100`). The final-only darkphoenix retrieval rerun completed
on two held-out seeds (`20260607`, `20260608`), 100 K-hop queries per target
size, no Glasgow. It confirms that MAG is not viable as a positive FullCov@50
result with the current objective: best observed dynamic FullCov is `16/200`
for 20-node queries, `2/200` for 50-node queries, and `0/200` for 100-node
queries at budget 100. Fixed retrieval is weaker (`K=50`: `2/200`, `0/200`,
`0/200` respectively). Keep MAG as a scalability stress/negative diagnostic,
not as a success headline. The safer positive third-dataset plan is a
homogeneous 500K--1M node graph or PubMed/Physics as near-term cross-dataset
evidence.

The locked per-query CSVs show why another lightweight stitching rule is
unlikely to rescue MAG. For target query sizes `20/50/100`, the average true
coarse footprint is `10.48/26.09/54.61` partitions and the median rank of the
last required partition is about `1296/1733/1879` out of 2,000. An oracle policy
that retrieves every partition up to the worst required rank would still cover
only `7/200`, `2/200`, and `0/200` queries by rank 100. At `K=50`, target-50
MAG queries have zero oracle FullCov under the current ranking. This is a
model/objective/representation failure, not merely a neighbor-expansion or
precision-at-small-k failure.

The active GraphSAGE+FullCov MAG diagnostic has not changed that conclusion as
of epoch 30: validation is `1/3/5` FullCov@20/50/100, below the completed MAG
GIN best `2/4/6` and roughly tied with the GIN final `2/3/6`. Do not block the
June 9 paper on this run.

Next MAG-only rescue smoke should change both representation and cache dynamics
rather than the retrieval selector. The implemented targeted test is
RGCN+FullCov with a momentum cache:

- `encoder_kind=rgcn`, so MAG edge types are used during message passing rather
  than only as static node features;
- typed reverse edges are preserved as separate relation IDs during MAG
  symmetrization;
- `momentum_cache_decay=0.99`, so partition-cache embeddings come from a slow
  EMA copy of the encoder instead of a rapidly stale snapshot;
- `coverage_topk=50`, because MAG target-50 queries average 26 true coarse
  partitions and the paper-critical budget is FullCov@50;
- `coverage_cvar_fraction=0.5`, so a larger fraction of the positive footprint
  receives worst-positive pressure;
- `max_live_positive_parts=64`, because MAG 100-node queries average 54.6 true
  partitions and the current cap of 24 is undersized;
- `cache_refresh_steps=10`, reducing stale negative/partition memory;
- `QueryTargetSizes="20,50,50,100,100"`, emphasizing the MAG cases that
  currently fail while still keeping 20-node queries in distribution.

The RGCN+momentum smoke was executed after migrating the hierarchy/checkpoint
into `hemachandraminchu` (`extra_wNzonK`). It completed 10 epochs but did not
beat the homogeneous GIN baseline at the practical budgets:

- final validation: `FullCov@20=1/100`, `@50=1/100`, `@100=2/100`;
- broad ranking improved only at high budgets: `@200=9/100`, `@500=19/100`,
  `@1000=42/100`;
- conclusion: RGCN+momentum does not by itself make MAG viable at K=50/100.

Executed launcher shape:

```powershell
.\scripts\run_graphsage_final_loss_modal.ps1 `
  -Dataset mag `
  -TrainingSeed 7201 `
  -Profile extra_wNzonK `
  -Resume `
  -Epochs 10 `
  -StepsPerEpoch 100 `
  -CoverageTopK 50 `
  -CoverageBucketSize 10 `
  -CoverageCvarFraction 0.5 `
  -MaxLivePositiveParts 64 `
  -CacheRefreshSteps 10 `
  -MaxTrainCoarseParts 80 `
  -QueryTargetSizes "20,50,50,100,100" `
  -QuerySizeJitter 5 `
  -EncoderKind rgcn `
  -MomentumCacheDecay 0.99 `
  -RunNameSuffix "mag_moco_topk50_live64_cvar50_e10"
```

The next same-partition rescue test was therefore overlap-aware candidate
assembly, not a change to the number of partitions. The implemented diagnostic
keeps `Coarse=2000` and the retrieved partition budget fixed, then tests
whether one-hop boundary nodes adjacent to the selected partitions contain all
query nodes. This is not hard partition FullCov; it is an exact-solver
candidate-containment test for overlapping boundary assembly.

Completed overlap diagnostic, 100 MAG 20-node k-hop queries, seed `20260607`,
GIN final/best only:

- fixed GIN-final K=50: hard partition FullCov `0/100`, overlap node FullCov
  `50/100`, overlap node recall `0.8175`;
- fixed GIN-final K=100: hard `1/100`, overlap `60/100`, recall `0.8640`;
- hybrid GIN-final B=50: hard `1/100`, overlap `50/100`, recall `0.8195`;
- hybrid GIN-final B=100: hard `5/100`, overlap `62/100`, recall `0.8615`;
- GIN-best is similar: fixed K=50 overlap `51/100`, hybrid B=100 overlap
  `62/100`.

This is the first MAG result that looks partially viable under the same 2,000
partition count, but the candidate-node upper bounds are large: about `467k`
nodes for fixed K=50 and `855k` nodes for fixed K=100 on GIN-final. The next
required engineering step is label/type-aware overlap pruning before Glasgow;
otherwise the solver candidate graph may be too large even when node
containment improves.

A follow-up signature-pruning diagnostic was run on MAG query target sizes
`20/50/100`, 100 queries each, seed `20260607`, using GIN-final only. The
signature filters are exact counters for now, not probabilistic Bloom filters.
They should be implemented later as Bloom/Roaring signatures once the token
family is chosen. Two useful token families emerged:

- `type_feat32`: safer production proxy using node type plus 32 signs from the
  feature vector. This needs only query features/types.
- `type_rel_feat32`: stronger exploratory proxy using node type, feature signs,
  and full-graph relation-degree buckets. It is useful diagnostically but should
  not be claimed as a clean query-only filter unless the query supplies the same
  typed relation context.

Key GIN-final overlap/signature tradeoffs:

| Query nodes | Retrieval | Budget | Hard partition FullCov | One-hop node FullCov | Overlap upper-bound nodes | `type_feat32` nodes | `type_rel_feat32` nodes |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | fixed | 50 | 0/100 | 50/100 | 467,464 | 117,797 | 76,105 |
| 20 | fixed | 100 | 1/100 | 60/100 | 854,869 | 229,305 | 145,591 |
| 20 | hybrid | 50 | 1/100 | 50/100 | 828,957 | 121,433 | 80,468 |
| 20 | hybrid | 100 | 5/100 | 62/100 | 1,348,576 | 233,345 | 152,145 |
| 50 | fixed | 50 | 0/100 | 32/100 | 471,260 | 133,067 | 100,940 |
| 50 | fixed | 100 | 0/100 | 38/100 | 843,259 | 252,629 | 189,899 |
| 50 | hybrid | 50 | 1/100 | 30/100 | 839,314 | 138,237 | 106,577 |
| 50 | hybrid | 100 | 1/100 | 36/100 | 1,363,482 | 256,767 | 196,417 |
| 100 | fixed | 50 | 0/100 | 18/100 | 495,441 | 145,359 | 113,823 |
| 100 | fixed | 100 | 0/100 | 24/100 | 868,053 | 273,917 | 213,849 |
| 100 | hybrid | 50 | 0/100 | 21/100 | 818,556 | 151,144 | 118,666 |
| 100 | hybrid | 100 | 0/100 | 34/100 | 1,342,976 | 278,320 | 218,982 |

Current best MAG operating points without changing the partition count:

- Smallest plausible solver candidate: fixed K=50 + one-hop overlap +
  `type_rel_feat32`, giving `50/100` containment for 20-node queries with about
  `76k` candidate nodes on average. The safer `type_feat32` version is about
  `118k` nodes.
- Highest 20-node containment at the tested budgets: hybrid B=100 + one-hop
  overlap + signature pruning, giving `62/100` containment with about `152k`
  (`type_rel_feat32`) or `233k` (`type_feat32`) candidate nodes.
- For 50-node queries, the best tested containment is only `38/100` fixed K=100
  or `36/100` hybrid B=100. For 100-node queries, the best tested containment is
  `34/100` hybrid B=100. These are diagnostics, not a positive MAG result yet.

The dynamic stitch variants did not beat fixed/hybrid overlap under this
diagnostic. The useful rescue path is not "more stitching" by itself; it is
retrieval plus one-hop overlap plus query-signature node pruning, followed by a
real solver test on the pruned candidate graph.

Retrieval-vs-filter controls were then added and run with the same MAG GIN-final
setup. This checks whether neural retrieval is actually doing useful work, or
whether the signature filter alone is enough. Controls:

- `filter_all_partitions`: skip retrieval and let the filter see all 2,000
  partitions. This gives `100/100` containment by construction but measures the
  residual candidate size after filtering the whole graph.
- `random_fixed`: choose a deterministic random set of 50 or 100 partitions, to
  separate retrieval quality from overlap/signature mechanics.

Result: retrieval is necessary. The filter-only control keeps full containment
but remains much too large, while random partitions miss almost everything.

| Query nodes | Method | Budget | One-hop node FullCov | `type_feat32` nodes | `type_rel_feat32` nodes | Retrieval ms/query | Signature ms/query |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | fixed retrieval | 50 | 50/100 | 117,797 | 76,105 | 6.39 | 2.22 |
| 20 | random | 50 | 2/100 | 100,721 | 60,688 | 0.00 | 2.68 |
| 20 | all partitions | 2000 | 100/100 | 4,008,024 | 2,408,182 | 0.00 | 90.62 |
| 50 | fixed retrieval | 50 | 32/100 | 133,067 | 100,940 | 6.15 | 3.94 |
| 50 | random | 50 | 0/100 | 104,634 | 75,236 | 0.00 | 4.37 |
| 50 | all partitions | 2000 | 100/100 | 4,110,110 | 2,948,043 | 0.00 | 155.15 |
| 100 | fixed retrieval | 50 | 18/100 | 145,359 | 113,823 | 6.32 | 6.50 |
| 100 | random | 50 | 0/100 | 106,429 | 81,573 | 0.00 | 6.71 |
| 100 | all partitions | 2000 | 100/100 | 4,223,700 | 3,228,769 | 0.00 | 244.09 |

Offline/index overheads from the same run:

- MAG homogeneous graph load: `1,939,743` nodes and `42,182,144` directed edges.
- One-hop overlap index: `1,560,767` boundary nodes and `12,664,593`
  partition-node overlap memberships across 2,000 partitions, built in `74.7s`.
- Average home partition size: `970` nodes.
- Average overlap memberships per partition: `6,332`; average home plus
  one-hop upper-bound per partition: `7,302` nodes.
- Signature indexes: all tested exact-counter signatures built in `27.3s`;
  `type_feat32` alone took `4.0s`, `type_rel_feat32` took `4.4s`.
- FAISS index over 2,000 partition embeddings: `12.5s`.

Conclusion: the filter is not a replacement for retrieval. It is a second-stage
candidate reducer. The promising MAG pipeline is retrieval first, overlap
second, signature pruning third. Without retrieval, signature pruning still
leaves millions of candidate nodes; with random retrieval, containment collapses.

### MAG Progressive Exact Cascade, Verified Fixed And Hybrid Retrieval

The first solver-backed MAG cascade is now verified on 100 locked 20-node
k-hop queries, seed `20260607`, using GIN-final, fixed neural budgets
`20,50,100`, one-hop partition overlap, `type_rel_feat32` signature pruning,
and exact query-label pruning before Glasgow. Output files:

- `runs/logs/overlap_cascade_mag_gin_fixed_sigrel32_label_q100_size20_seed20260607_summary.csv`
- `runs/logs/overlap_cascade_mag_gin_fixed_sigrel32_label_q100_size20_seed20260607_per_query.csv`

Fixed result:

| Metric | Value |
|---|---:|
| Solved total | `49/100` |
| Solved at budget 20 | `32/100` |
| Additional solved at budget 50 | `11/100` |
| Additional solved at budget 100 | `6/100` |
| Unsolved | `51/100` |
| Pruned candidate FullCov rows | `55/100` |
| Solver timeouts | `30` |
| Avg retrieval time/query | `0.0115s` |
| Avg candidate build+prune/query | `8.45s` |
| Avg solver time/query | `7.41s` |
| Avg overlap nodes | `248,747` |
| Avg signature nodes | `47,966` |
| Avg final label-pruned nodes | `20,412` |

This changes the MAG story but does not make MAG solved. Retrieval-only hard
partition FullCov is still poor; the rescue comes from overlap containment plus
strong exact pruning. The result is paper-useful as a progressive verification
diagnostic:

1. neural retrieval matters (`random K=50` only contains `2/100` 20-node
   queries);
2. filter-only is not enough (`all partitions` contains `100/100` but leaves
   about `2.4M` candidate nodes);
3. retrieval+overlap+signature+label pruning gives a workable 20-node cascade
   on roughly `20k` candidates/query;
4. 50- and 100-node MAG queries remain weak and should not be claimed as
   successful.

Hybrid result is now complete and is the stronger MAG cascade row:

- `runs/logs/overlap_cascade_mag_gin_hybrid_sigrel32_label_q100_size20_seed20260607_summary.csv`
- `runs/logs/overlap_cascade_mag_gin_hybrid_sigrel32_label_q100_size20_seed20260607_per_query.csv`

| Metric | Value |
|---|---:|
| Solved total | `54/100` |
| Solved at budget 20 | `33/100` |
| Additional solved at budget 50 | `12/100` |
| Additional solved at budget 100 | `9/100` |
| Unsolved | `46/100` |
| Pruned candidate FullCov rows | `60/100` |
| Solver timeouts | `26` |
| Avg retrieval time/query | `0.0076s` |
| Avg candidate build+prune/query | `7.48s` |
| Avg solver time/query | `7.09s` |
| Avg overlap nodes | `338,036` |
| Avg signature nodes | `51,067` |
| Avg final label-pruned nodes | `21,659` |

Use the hybrid row for the strongest MAG statement. Keep the fixed row as a
cleaner ablation/control. This still does not rescue 50-/100-node MAG queries.

### Final Cascade Checks: Arxiv And Cora

Arxiv fixed cascade, final local model, 100 locked 20-node K-hop queries:

- `runs/logs/overlap_cascade_arxiv_final_fixed_sigfeat32_label_q100_size20_seed20260607_summary.csv`
- `runs/logs/overlap_cascade_arxiv_final_fixed_sigfeat32_label_q100_size20_seed20260607_per_query.csv`

| Metric | Value |
|---|---:|
| Solved total | `99/100` |
| Solved at budget 20 | `85/100` |
| Additional solved at budget 50 | `9/100` |
| Additional solved at budget 100 | `5/100` |
| Unsolved | `1/100` |
| Pruned candidate FullCov rows | `99/100` |
| Solver timeouts | `0` |
| Avg retrieval time/query | `0.0050s` |
| Avg candidate build+prune/query | `0.108s` |
| Avg solver time/query | `0.0247s` |
| Avg overlap nodes | `45,211` |
| Avg signature nodes | `82.6` |
| Avg final label-pruned nodes | `23.3` |

Cora fixed cascade, existing model, no retraining, 100 queries for each target
size `20/50/100`:

- `runs/logs/overlap_cascade_cora_fixed_sigfeat32_label_q100_sizes20_50_100_seed20260607_summary.csv`
- `runs/logs/overlap_cascade_cora_fixed_sigfeat32_label_q100_sizes20_50_100_seed20260607_per_query.csv`

| Query nodes | Solved | Solved at K=20 | Avg retrieval | Avg candidate build+prune | Avg solver | Avg final nodes |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | `100/100` | `100/100` | `0.0061s` | `0.951s` | `0.0365s` | `20.6` |
| 50 | `100/100` | `100/100` | `0.0062s` | `0.971s` | `0.0232s` | `52.4` |
| 100 | `100/100` | `100/100` | `0.0069s` | `0.983s` | `0.0331s` | `103.7` |

Cora remains a sanity/timing result because there are only 20 coarse
partitions, so K=20 already includes every partition.

K=20 remains a strict reporting budget, but it should not be the only MAG
training target. For 50- and 100-node MAG queries the true coarse footprint
often exceeds 20, so FullCov@20 is structurally impossible for many rows.
Training with `coverage_topk=50` still reports K=20 validation but gives the
objective a budget aligned with the FullCov@50 target we actually need.

Local MAG retrieval aggregate:

- `runs/logs/mag_retrieval_locked_aggregate.md`
- `runs/logs/retrieval_mag_khop_q100_sizes20_50_100_seed20260607_summary.csv`
- `runs/logs/retrieval_mag_khop_q100_sizes20_50_100_seed20260608_summary.csv`
- `runs/logs/retrieval_mag_khop_sigprune32_allq_gin_q100_sizes20_50_100_seed20260607_summary.csv`
- `runs/logs/retrieval_mag_khop_sigprune32_allq_gin_q100_sizes20_50_100_seed20260607_per_query.csv`
- `runs/logs/retrieval_mag_khop_sigcontrol_gin_q100_sizes20_50_100_seed20260607_summary.csv`
- `runs/logs/retrieval_mag_khop_sigcontrol_gin_q100_sizes20_50_100_seed20260607_per_query.csv`

Yelp was also smoke-tested in the `darkphoenix696969696969` Modal workspace
with the same cached-negative training path, but it is too slow for the June 9
submission: the smoke run reached only step `6/10` of epoch 1 after about
`15m52s`. The run was stopped and Yelp should be excluded from the paper rather
than reported as incomplete evidence.

Minimum defensible ECML submission package if time is tight:

1. Replace the abstract/results claims with FullCov-aware language.
2. Add the matched-budget Arxiv objective ablation.
3. Add fixed Arxiv retrieval tables with FullCov primary. Keep dynamic
   retrieval optional/appendix unless it materially improves the headline
   fixed-K story.
4. Keep Cora end-to-end Glasgow as the exact-solver speedup sanity check.
5. Add one cross-dataset retrieval table or explicitly label MAG as a
   scalability diagnostic if its retrieval CSVs remain weak or incomplete.
6. Move exploratory MC dropout, MCTS, fine-coverage loss, node loss, and
   multi-view reranking to appendix/negative ablations unless new locked
   results justify them.
7. Render the paper PDF and visually check page count, table overflow, figure
   readability, and LNCS margins before calling any version submission-ready.
   MiKTeX is now installed locally and should be the default render path; use
   Overleaf or a render container only if local compilation fails.

## 6. Regenerate Tables And Figures

After copying approved Modal result CSVs into a new benchmark release folder, regenerate summaries and figures.

```powershell
python scripts/summarize_paper_benchmarks.py --input-dir benchmarks/paper_results --output paper/benchmark_summary.md
python scripts/generate_paper_plots.py
```

## 7. Paper Changes Required

- Replace claims of finding all subgraph isomorphisms with bounded-region exact verification.
- State that Glasgow is exact inside the stitched candidate region.
- Report K-hop separately as a bounded retrieval stress test.
- Add ablation tables for coverage strength and K.
- Add at least one additional dataset table or clearly labeled probe.
- Add architecture and hyperparameter tables.
- Add reviewer-requested baseline discussion, including what was reproduced and what was not.
