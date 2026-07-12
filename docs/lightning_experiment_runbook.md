# Lightning Experiment Runbook (staged for compute)

Everything needed to launch/process the remaining experiments. Resolved config and
exact commands so any compute session is one command per experiment.

## Resolved Lightning config
- Account/owner: **whenthedarknightrises** (user, not org). Teamspace: **financial-llm-training-project**.
- Creds: set `LIGHTNING_USER_ID` + `LIGHTNING_API_KEY` (the whenthedarknightrises account); the
  SDK reads `~/.lightning/credentials.json` — `lightning login` rewrites it. (swastik9895 backup at
  `~/.lightning/credentials.json.bak_swastik`.)
- Launcher: `scripts/lightning_mag_benchmark.py launch-job` (Windows shim `scripts/lightning_cli_windows.py`).
- Machine **CPU_X_8**, cloud **gcp-lightning-public-prod**, `--interruptible` (GCP spot, cheapest).
- MAG package (has hierarchy + best/final RGCN models + raw data; generates queries on-node):
  **`jigsaw-mag-rgcn-benchmark-package-connected-v3`**.
- Code overlay (my selective/feature_index/filterall_raw methods): **`jigsaw-mag-pareto-overlay-v2`**
  (rebuild from `runs/code_patch_selective_overlap/` via `upload_model(...overlay-vN, path=...)`).
- Methods available (via the overlay's `run_mag_benchmark_matrix_local.py`): neural_component,
  mean_rrf_component, mean_feature_component, topo_feature_component, random_component,
  filterall_component, **neural_selective** (`--overlap-max-parts 8 --overlap-label-compatible`),
  **neural_selective_topk** (`--overlap-max-parts 8`), **feature_index_component** (classical index),
  **filterall_raw** (no-retrieval full-graph Glasgow).

## Launched (2026-06-22)
| Job | Methods | Result model | Status |
|---|---|---|---|
| jigsaw-mag-pareto-selective-gcp-cpux8-v1 | neural_component, neural_selective, neural_selective_topk | jigsaw-mag-pareto-selective-v1-results | Running |
| jigsaw-mag-baseline-fullglasgow-gcp-cpux8-v1 | filterall_raw (15 q) | jigsaw-mag-baseline-fullglasgow-v1-results | Done: 0/15 solved, all timeout |
| jigsaw-mag-featureindex-baseline-gcp-cpux8-v1 | feature_index_component | jigsaw-mag-featureindex-baseline-v1-results | Running |

## Launch template
```bash
source <scratch>/lenv.sh   # sets LIGHTNING_USER_ID/API_KEY
.venv_modal/Scripts/python.exe scripts/lightning_mag_benchmark.py launch-job \
  --owner whenthedarknightrises --teamspace financial-llm-training-project \
  --package-model jigsaw-mag-rgcn-benchmark-package-connected-v3 \
  --code-patch-model jigsaw-mag-pareto-overlay-v2 \
  --result-model <NEW-RESULT-MODEL> --job-name <JOB-NAME> \
  --machine CPU_X_8 --cloud gcp-lightning-public-prod --interruptible \
  --methods "<METHODS>" --query-types positive --seeds 20260607,20260608 \
  --queries 50 --target-sizes 20,50,100 --budgets 20,50,100,200,500,1000 \
  --solver-timeout 5 --workers 2 --parallel-mode task
```

## Remaining experiments (ready to launch)
1. **More seeds (CIs):** add `--seeds 20260609,20260610` for neural_component + neural_selective +
   the baselines; widens the bootstrap CIs / paired tests in `tab:significance`.
2. **Walk-aware retrain (random_walk fix):** `bash scripts/launchers/run_mag_walk_aware_retrain.sh`
   on the training box (see `docs/walk_aware_retrain_runbook.md`); then re-benchmark the new
   checkpoint with `--methods neural_component --query-types random_walk` and compare coverage-by-budget.
3. **DBLP-HetG / KG dataset (relation-rich hetero):** needs a loader in `src/data.py` + a package
   build; then the same launch template with `--dataset dblp`. The strongest scale/diversity add.
4. **ogbn-papers100M:** billion-edge scale point; package build + launch.

## Processing (after any job)
```python
from lightning_sdk.models import download_model
download_model("whenthedarknightrises/financial-llm-training-project/<RESULT-MODEL>", "runs/<dst>")
# then: scripts/summarize_production_benchmarks.py runs/<dst>/**/*_per_query.csv --output <summary>
# merge into benchmarks/paper_results/final_results/ via /tmp/merge.py pattern (substitute rows)
# figures: scripts/generate_submission_figures.py ; significance: scripts/analyze_significance.py
```

## Pareto processing (when jigsaw-mag-pareto-selective-v1 lands)
Compare within-job neural_component (blunt) vs neural_selective vs neural_selective_topk on
(positive solve rate, avg_total_s, avg_candidate_nodes). Build the recall-vs-latency and
recall-vs-candidate-size Pareto (selective should match blunt recall at lower latency/size; plot
vs the FilterAll 98.4%/5.85s ceiling). Add as `fig_pareto_mag.png` + a paragraph in the MAG results.
