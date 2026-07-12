# GNN-PE Feasibility Spike

GNN-PE is the closest neural-plus-exact related work, but integrating it with
Jigsaw is a format/runtime spike rather than a normal Jigsaw benchmark launch.
The public workflow is:

1. Prepare the dataset with `GNN-PE/gnnpe.py`.
2. Build the C++17 engine with CMake.
3. Run `main` once in offline mode to build the path index.
4. Run `main` in online mode for each query.

The public README requires Linux, GCC/G++ >= 12.2, and CMake >= 3.28. The local
Windows environment currently does not satisfy that build stack, so the spike
should run in a Linux environment such as Lightning.

## What We Test First

Start with Cora, not MAG. The first question is whether GNN-PE can ingest our
large connected planted queries at all. Cora is small enough that failures are
format/query-regime failures rather than cloud-cost problems.

The spike exports:

- `data_graph.gpickle.gz`: NetworkX graph for GNN-PE's Python prep.
- `data_graph.graph`: C++ graph format (`t`, `v`, `e` lines).
- `queries/*.graph`: small BFS probes plus Jigsaw-style 20/50/100-node queries.
- `manifest.json`: query metadata.
- `run_gnnpe_spike.sh`: Linux commands to prep/build/offline/online-test.

By default online queries use `-n 1`, so the first pass measures ingestibility
and first-match latency instead of enumerating all embeddings.

## Export Command

Run this in a Linux environment with the Jigsaw dependencies installed:

```bash
python scripts/analysis/gnnpe_feasibility_spike.py \
  --dataset cora \
  --data-root data \
  --hierarchy runs/overlap_models/cora/cora_hierarchies_finecov_v1.pt \
  --gnnpe-repo runs/external/GNN-PE \
  --output runs/gnnpe_spike/cora \
  --query-types small_bfs,k_hop,single,multi_fine,multi_coarse,random_walk \
  --target-sizes 8,20,50,100 \
  --queries-per-cell 1 \
  --answer-limit 1 \
  --timeout-seconds 300
```

Then run:

```bash
bash runs/gnnpe_spike/cora/run_gnnpe_spike.sh
```

## Interpreting Outcomes

- If small 8-node probes work but 20/50/100-node queries time out or fail, the
  honest result is that GNN-PE and Jigsaw cover different query regimes.
- If 20/50/100-node Cora queries work, extend to Arxiv before MAG.
- If Cora export/prep fails, fix the converter first; do not spend GPU/CPU
  budget on larger graphs.
- If all Cora queries work quickly, the fairer paper comparison is still scoped:
  GNN-PE is globally complete, while Jigsaw is budgeted and exact only inside
  the retrieved candidate region.

This spike is not evidence that Jigsaw beats GNN-PE. It only answers whether a
fair head-to-head is operationally feasible for our planted connected workload.
