# Query Generator Alignment

This note records the active query semantics after the cleanup pass.

## Canonical Paper Benchmark

`scripts/benchmark_glasgow.py` now tags rows with:

```text
query_generator_version=aligned_connected_v2
```

The active benchmark query families are:

- `single`: connected fragment sampled inside one coarse partition.
- `multi_fine`: connected fragments sampled from multiple fine partitions under the same coarse partition.
- `multi_coarse`: connected fragments sampled from fine partitions that span neighboring coarse partitions; when fine partitions are available, it starts from an actual fine-level bridge across a coarse edge.
- `k_hop`: connected BFS blob inside a 3-hop neighborhood, matching the training k-hop semantics.

The old benchmark k-hop generator was useful as a hard stress diagnostic, but it was not fully aligned with training: it grew `k=1..4`, could cap by tensor order, and could produce broader stitched contexts than the model was trained to retrieve.

## Training Semantics

`scripts/train_jigsaw_model.py` remains the canonical training script. Its k-hop strategy samples a connected BFS blob inside a fixed 3-hop neighborhood, then uses all touched coarse partitions as `coverage_coarse_ids` for the all-positive partition coverage loss. Context graphs are still bounded for memory.

The v3 training path also records `coverage_fine_ids` for the fine partitions touched by each query and can optimize them with `--gamma-fine-partition`. This is default-off for backward compatibility; the active fine-coverage run sets `--gamma-fine-partition 0.5`.

The current `coverage_v2_allpos_fresh` Modal run is therefore not invalidated by this benchmark cleanup.

## Legacy Utility Generator

`src/query_generator.py` already uses connected fragments for the older evaluator path. Its k-hop query is also a BFS-trimmed connected query inside a k-hop neighborhood, so it is semantically closer to training than the previous Glasgow benchmark implementation was.

## Paper Reporting

For final tables, report:

- query generator version,
- true coarse partition count avg/max,
- FullCov@K / retrieval complete rate,
- coarse recall at `k` as a secondary diagnostic,
- Glasgow found rate,
- stitched node count,
- oracle found rate for k-hop diagnostics.

For fine-boundary candidate construction, also report candidate FullCov, fine
candidate FullCov, pre-prune node count, pruned node count, and whether label
pruning was enabled.
