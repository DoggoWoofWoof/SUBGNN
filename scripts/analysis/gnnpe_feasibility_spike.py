"""Export a small Jigsaw workload for a GNN-PE feasibility spike.

This script does not modify GNN-PE. It writes the file formats expected by
https://github.com/JamesWhiteSnow/GNN-PE:

  - data_graph.gpickle for the Python partition/prep step
  - data_graph.graph for the C++ offline/online engine
  - queries/*.graph for online query tests
  - run_gnnpe_spike.sh with the Linux commands to build/run the spike

The intended first target is Cora. Cora is small enough to expose format and
query-regime problems before spending time on Arxiv/MAG.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import networkx as nx
import torch

import benchmark_glasgow as bench
from benchmark_overlap_glasgow_cascade import generate_cascade_queries
from benchmark_retrieval import prepare_hierarchy
from src.data import build_single_hierarchy, load_dataset
from src.utils import feature_to_label


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_strings(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _undirected_edges(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for u, v in edge_index.detach().cpu().long().t().tolist():
        if u == v:
            continue
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        pairs.add((a, b))
    return sorted(pairs)


def _contiguous_labels(data, source: str) -> list[int]:
    raw = []
    if source == "class":
        y = getattr(data, "y", None)
        if y is None:
            raise ValueError("--label-source class requested, but dataset has no y labels")
        y_cpu = y.detach().cpu()
        if y_cpu.ndim > 1 and y_cpu.size(-1) > 1:
            y_cpu = y_cpu.argmax(dim=-1)
        raw = [int(v) for v in y_cpu.view(-1).long().tolist()]
    elif source == "feature":
        if getattr(data, "x", None) is None:
            raise ValueError("--label-source feature requested, but dataset has no x features")
        x_cpu = data.x.detach().cpu()
        for idx in range(data.num_nodes):
            raw.append(int(feature_to_label(x_cpu[idx])))
    elif source == "node_type":
        if not hasattr(data, "node_type"):
            raise ValueError("--label-source node_type requested, but dataset has no node_type")
        raw = [int(v) for v in data.node_type.detach().cpu().long().tolist()]
    elif source == "constant":
        raw = [0 for _ in range(data.num_nodes)]
    else:
        raise ValueError(f"Unsupported label source: {source}")

    label_map = {value: i for i, value in enumerate(sorted(set(raw)))}
    return [label_map[value] for value in raw]


def _degrees(num_nodes: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    deg = [0 for _ in range(num_nodes)]
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def write_graph(path: Path, num_nodes: int, edges: Sequence[tuple[int, int]], labels: Sequence[int]) -> None:
    deg = _degrees(num_nodes, edges)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"t {num_nodes} {len(edges)}\n")
        for node_id in range(num_nodes):
            handle.write(f"v {node_id} {int(labels[node_id])} {int(deg[node_id])}\n")
        for u, v in edges:
            handle.write(f"e {int(u)} {int(v)}\n")


def write_gpickle(path: Path, num_nodes: int, edges: Sequence[tuple[int, int]], labels: Sequence[int]) -> None:
    graph = nx.Graph()
    graph.add_nodes_from((i, {"label": int(labels[i])}) for i in range(num_nodes))
    graph.add_edges_from((int(u), int(v)) for u, v in edges)
    with path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _adjacency(edges: Sequence[tuple[int, int]], num_nodes: int) -> list[list[int]]:
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs_nodes(adj: Sequence[Sequence[int]], target_size: int, rng: random.Random) -> list[int] | None:
    starts = list(range(len(adj)))
    rng.shuffle(starts)
    for start in starts[:200]:
        seen = {start}
        order = [start]
        queue = deque([start])
        while queue and len(order) < target_size:
            cur = queue.popleft()
            nbrs = list(adj[cur])
            rng.shuffle(nbrs)
            for nbr in nbrs:
                if nbr in seen:
                    continue
                seen.add(nbr)
                order.append(nbr)
                queue.append(nbr)
                if len(order) >= target_size:
                    break
        if len(order) >= target_size:
            return order[:target_size]
    return None


def _induced_query_edges(global_nodes: Sequence[int], edge_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    local = {int(node): i for i, node in enumerate(global_nodes)}
    out: list[tuple[int, int]] = []
    for i, u in enumerate(global_nodes):
        for v in global_nodes[i + 1 :]:
            a, b = (int(u), int(v)) if u < v else (int(v), int(u))
            if (a, b) in edge_set:
                out.append((local[int(u)], local[int(v)]))
    return sorted(out)


def _write_query(
    path: Path,
    global_nodes: Sequence[int],
    data_labels: Sequence[int],
    edge_set: set[tuple[int, int]],
) -> dict:
    global_nodes = [int(node) for node in global_nodes]
    query_edges = _induced_query_edges(global_nodes, edge_set)
    query_labels = [int(data_labels[node]) for node in global_nodes]
    write_graph(path, len(global_nodes), query_edges, query_labels)
    return {
        "path": str(path),
        "nodes": len(global_nodes),
        "edges": len(query_edges),
        "global_nodes": global_nodes,
        "label_count": len(set(query_labels)),
    }


def _load_or_build_hierarchy(dataset: str, data, hierarchy_path: Path | None, output_dir: Path):
    if hierarchy_path and hierarchy_path.exists():
        hierarchy = _load_torch(hierarchy_path)
    else:
        coarse, fine = {"cora": (20, 5), "arxiv": (200, 5), "mag": (2000, 5)}.get(
            dataset, (20, 5)
        )
        hierarchy = build_single_hierarchy(data, coarse, fine)
        torch.save(hierarchy, output_dir / f"{dataset}_hierarchies_finecov_v1.pt")
    # Overlap-model runs save a LIST of hierarchies (all_hierarchies); prepare_hierarchy wants the
    # single hierarchy dict, so unwrap a list/tuple to its first element. Fixes the v3/v4 failure
    # (ValueError: dictionary update sequence element #0 has length 10; 2 is required).
    if isinstance(hierarchy, (list, tuple)):
        hierarchy = hierarchy[0]
    return prepare_hierarchy(data, hierarchy)


def export_spike(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    output_dir = Path(args.output).resolve()
    query_dir = output_dir / "queries"
    output_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset, root=args.data_root)
    data = data.cpu()
    edges = _undirected_edges(data.edge_index)
    edge_set = set(edges)
    labels = _contiguous_labels(data, args.label_source)
    adj = _adjacency(edges, data.num_nodes)

    write_graph(output_dir / "data_graph.graph", data.num_nodes, edges, labels)
    write_gpickle(output_dir / "data_graph.gpickle", data.num_nodes, edges, labels)

    hierarchy = None
    requested_types = _parse_strings(args.query_types)
    if any(qt != "small_bfs" for qt in requested_types):
        hierarchy = _load_or_build_hierarchy(
            args.dataset,
            data,
            Path(args.hierarchy) if args.hierarchy else None,
            output_dir,
        )

    manifest = {
        "dataset": args.dataset,
        "nodes": int(data.num_nodes),
        "edges_undirected": int(len(edges)),
        "label_source": args.label_source,
        "label_count": int(len(set(labels))),
        "queries": [],
        "gnnpe": {
            "repo": str(Path(args.gnnpe_repo).resolve()),
            "partition_num": int(args.partition_num),
            "path_length": int(args.path_length),
            "embedding_dim": int(args.embedding_dim),
            "answer_limit": args.answer_limit,
        },
    }

    target_sizes = _parse_ints(args.target_sizes)
    query_index = 0
    for target_size in target_sizes:
        for qtype in requested_types:
            if qtype == "small_bfs":
                for _ in range(args.queries_per_cell):
                    nodes = _bfs_nodes(adj, target_size, rng)
                    if nodes is None:
                        continue
                    qpath = query_dir / f"q{query_index:04d}_{qtype}_n{target_size}.graph"
                    row = _write_query(qpath, nodes, labels, edge_set)
                    row.update({"query_type": qtype, "target_size": target_size})
                    manifest["queries"].append(row)
                    query_index += 1
                continue

            if hierarchy is None:
                hierarchy = _load_or_build_hierarchy(
                    args.dataset,
                    data,
                    Path(args.hierarchy) if args.hierarchy else None,
                    output_dir,
                )
            generated = generate_cascade_queries(
                data,
                hierarchy,
                args.queries_per_cell,
                target_size,
                args.seed + 1009 * target_size + query_index,
                qtype,
            )
            for item in generated:
                nodes = [int(node) for node in item["query_nodes"].detach().cpu().long().tolist()]
                qpath = query_dir / f"q{query_index:04d}_{qtype}_n{target_size}.graph"
                row = _write_query(qpath, nodes, labels, edge_set)
                row.update({"query_type": qtype, "target_size": target_size})
                manifest["queries"].append(row)
                query_index += 1

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_runner(output_dir, args)
    print(
        f"[GNN-PE SPIKE] wrote {len(manifest['queries'])} queries and data graph to {output_dir}",
        flush=True,
    )


def write_runner(output_dir: Path, args: argparse.Namespace) -> None:
    gnnpe_repo = Path(args.gnnpe_repo).resolve()
    gnnpe_src = gnnpe_repo / "GNN-PE"
    script = output_dir / "run_gnnpe_spike.sh"
    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "$0")" && pwd)"
GNNPE_REPO="${{GNNPE_REPO:-{gnnpe_repo.as_posix()}}}"
GNNPE_SRC="${{GNNPE_SRC:-{gnnpe_src.as_posix()}}}"
PARTITIONS="${{PARTITIONS:-{args.partition_num}}}"
PATH_LENGTH="${{PATH_LENGTH:-{args.path_length}}}"
EMBED_DIM="${{EMBED_DIM:-{args.embedding_dim}}}"
ANSWER_LIMIT="${{ANSWER_LIMIT:-{args.answer_limit}}}"
TIMEOUT_SECONDS="${{TIMEOUT_SECONDS:-{args.timeout_seconds}}}"

python "$GNNPE_SRC/gnnpe.py" --f "$SPIKE_DIR/" --d "$SPIKE_DIR/data_graph.gpickle" --p "$PARTITIONS" --l "$PATH_LENGTH"

cmake -S "$GNNPE_SRC" -B "$SPIKE_DIR/gnnpe_build" -DCMAKE_CXX_FLAGS="-march=x86-64" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$SPIKE_DIR/gnnpe_build" -j"$(nproc)"

"$SPIKE_DIR/gnnpe_build/src/main" -f "$SPIKE_DIR/" -d "$SPIKE_DIR/data_graph.graph" -m offline -p "$PARTITIONS" -l "$PATH_LENGTH" -e "$EMBED_DIM"

mkdir -p "$SPIKE_DIR/online_logs"

# Self-test on GNN-PE's own bundled Test data (known-good input). If this prints a positive
# Answer Number but our planted queries print 0, the 0-match issue is our export/graph; if this
# is also 0, the issue is the build/binary/params. Localizes the v10 zero-answers finding.
TESTDIR="$GNNPE_SRC/../Test"
if [ -f "$TESTDIR/data_graph.graph" ] && [ -f "$TESTDIR/query_graph.graph" ]; then
  echo "[GNN-PE SELFTEST] prep+offline+online on bundled Test data"
  python "$GNNPE_SRC/gnnpe.py" --f "$TESTDIR/" --d "$TESTDIR/data_graph.gpickle.gz" --p 5 --l 2 > "$SPIKE_DIR/online_logs/_selftest_prep.log" 2>&1 || true
  "$SPIKE_DIR/gnnpe_build/src/main" -f "$TESTDIR/" -d "$TESTDIR/data_graph.graph" -m offline -p 5 -l 2 -e 2 > "$SPIKE_DIR/online_logs/_selftest_offline.log" 2>&1 || true
  "$SPIKE_DIR/gnnpe_build/src/main" -f "$TESTDIR/" -d "$TESTDIR/data_graph.graph" -q "$TESTDIR/query_graph.graph" -m online -p 5 -l 2 -e 2 -n 0 > "$SPIKE_DIR/online_logs/_selftest_online.log" 2>&1 || true
  echo "[GNN-PE SELFTEST] $(grep -i 'Answer Number' "$SPIKE_DIR/online_logs/_selftest_online.log" | tail -1)"
fi

for query in "$SPIKE_DIR"/queries/*.graph; do
  name="$(basename "$query" .graph)"
  echo "[GNN-PE ONLINE] $name"
  timeout "$TIMEOUT_SECONDS" "$SPIKE_DIR/gnnpe_build/src/main" \\
    -f "$SPIKE_DIR/" \\
    -d "$SPIKE_DIR/data_graph.graph" \\
    -q "$query" \\
    -m online \\
    -p "$PARTITIONS" \\
    -l "$PATH_LENGTH" \\
    -e "$EMBED_DIM" \\
    -n "$ANSWER_LIMIT" \\
    > "$SPIKE_DIR/online_logs/${{name}}.log" 2>&1 || echo "[TIMEOUT/FAIL] $name"
done

echo "[GNN-PE CLEANUP] removing bulky rebuildable artifacts"
rm -rf "$SPIKE_DIR/gnnpe_build" "$SPIKE_DIR/gnn-pe" "$SPIKE_DIR/data_graph.gpickle"
find "$SPIKE_DIR" -maxdepth 1 -type f \\( \\
  -name '*.idx' -o \\
  -name '*.index' -o \\
  -name '*.dat' -o \\
  -name '*.bin' -o \\
  -name '*.npy' -o \\
  -name '*.npz' \\
\\) -delete
""",
        encoding="utf-8",
    )
    script.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cora", choices=["cora", "arxiv", "mag"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs/gnnpe_spike/cora")
    parser.add_argument("--hierarchy", default="runs/overlap_models/cora/cora_hierarchies_finecov_v1.pt")
    parser.add_argument("--gnnpe-repo", default="runs/external/GNN-PE")
    parser.add_argument("--query-types", default="small_bfs,k_hop,single,multi_fine,multi_coarse,random_walk")
    parser.add_argument(
        "--label-source",
        default="class",
        choices=["class", "feature", "node_type", "constant"],
        help=(
            "Node labels exported to GNN-PE. Class labels are the default for Cora because "
            "feature-hash labels make almost every Cora node unique and caused near-zero "
            "GNN-PE matches despite planted queries."
        ),
    )
    parser.add_argument("--target-sizes", default="8,20,50,100")
    parser.add_argument("--queries-per-cell", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--partition-num", type=int, default=20)
    parser.add_argument("--path-length", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=2)
    parser.add_argument("--answer-limit", default="1")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    export_spike(args)


if __name__ == "__main__":
    main()
