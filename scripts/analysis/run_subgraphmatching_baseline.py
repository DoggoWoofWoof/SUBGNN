"""
Scoped external exact-matcher probe using the SIGMOD'20 SubgraphMatching binary.

This is intentionally narrow: generate the same planted connected query families
used by the overlap-cascade benchmark, export the full target graph plus queries
in SubgraphMatching format, and run a small set of classical filter/order/engine
presets (CFL, DPiso, GQL/RI). It is meant to answer "can these established
in-memory matchers run on our protocol at all?" without producing large artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_overlap_glasgow_cascade as cascade  # noqa: E402
from src.utils import feature_to_label  # noqa: E402


PRESETS = {
    # Recommended combinations from the SubgraphMatching README.
    "cfl": {"filter": "CFL", "order": "CFL", "engine": "LFTJ"},
    "dpiso": {"filter": "DPiso", "order": "DPiso", "engine": "DPiso"},
    "gql": {"filter": "GQL", "order": "GQL", "engine": "LFTJ"},
    "ri": {"filter": "GQL", "order": "RI", "engine": "LFTJ"},
    "turboiso": {"filter": "TSO", "order": "TSO", "engine": "LFTJ"},
}


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(part) for part in parse_csv(value)]


def class_labels(data) -> torch.Tensor:
    if not hasattr(data, "y") or data.y is None:
        return torch.zeros(int(data.num_nodes), dtype=torch.long)
    y = data.y.detach().cpu().view(-1).long()
    node_type = getattr(data, "node_type", None)
    if node_type is None:
        node_type = torch.zeros(int(data.num_nodes), dtype=torch.long)
    else:
        node_type = node_type.detach().cpu().view(-1).long()
    base = int(y[y >= 0].max().item()) + 1 if bool((y >= 0).any()) else 1
    within_type = torch.where(y >= 0, y, torch.zeros_like(y))
    return node_type * base + within_type


def feature_labels(data) -> torch.Tensor:
    if not hasattr(data, "x") or data.x is None:
        return torch.zeros(int(data.num_nodes), dtype=torch.long)
    return torch.tensor(
        [feature_to_label(data.x[i]) for i in range(int(data.num_nodes))],
        dtype=torch.long,
    )


def _feature_hash_int(vector) -> int:
    if vector is None:
        return 0
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().cpu().numpy()
    if np.all(np.isin(vector, [0, 1])):
        feats_tuple = tuple(np.where(vector == 1)[0].tolist())
    else:
        feats_tuple = tuple(np.round(vector, 4).tolist())
    return int(hashlib.md5(str(feats_tuple).encode("utf-8")).hexdigest(), 16)


def feature_bucket_labels(data, bucket_count: int) -> torch.Tensor:
    if bucket_count <= 0:
        raise ValueError(f"feature bucket count must be positive, got {bucket_count}")
    if not hasattr(data, "x") or data.x is None:
        return torch.zeros(int(data.num_nodes), dtype=torch.long)
    return torch.tensor(
        [_feature_hash_int(data.x[i]) % bucket_count for i in range(int(data.num_nodes))],
        dtype=torch.long,
    )


def labels_for_source(data, label_source: str) -> torch.Tensor:
    if label_source == "feature":
        return feature_labels(data)
    if label_source.startswith("feature_bucket_"):
        return feature_bucket_labels(data, int(label_source.rsplit("_", 1)[-1]))
    if label_source == "class":
        return class_labels(data)
    if label_source == "zero":
        return torch.zeros(int(data.num_nodes), dtype=torch.long)
    raise ValueError(f"Unknown label source: {label_source}")


def write_subgraphmatching_graph(data, path: Path, labels: Optional[torch.Tensor] = None) -> Dict[str, int]:
    """Write the vertex-labeled graph format expected by SubgraphMatching.out."""
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.detach().cpu()
    edges = set()
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        src, dst = int(src), int(dst)
        if src == dst:
            continue
        edges.add((min(src, dst), max(src, dst)))
    degree = [0] * num_nodes
    for src, dst in edges:
        degree[src] += 1
        degree[dst] += 1
    if labels is None:
        labels = getattr(data, "node_label", None)
    if labels is None:
        labels = torch.zeros(num_nodes, dtype=torch.long)
    labels = labels.detach().cpu().view(-1).long()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"t {num_nodes} {len(edges)}\n")
        for node in range(num_nodes):
            handle.write(f"v {node} {int(labels[node].item())} {degree[node]}\n")
        for src, dst in sorted(edges):
            handle.write(f"e {src} {dst}\n")
    return {"nodes": num_nodes, "edges": len(edges), "label_count": int(labels.unique().numel())}


def parse_embedding_count(stdout: str) -> int:
    patterns = [
        r"#Embeddings:\s*(\d+)",
        r"Embedding Cnt:\s*(\d+)",
        r"Enumerate\s+(\d+)\s+results",
        r"Total\s+embeddings:\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    # Some patched builds print one "Embedding N:" line per result.
    return sum(1 for line in stdout.splitlines() if line.strip().lower().startswith("embedding "))


def parse_solver_metrics(stdout: str) -> Dict[str, object]:
    def first_float(pattern: str):
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        return float(match.group(1)) if match else ""

    core_values = [int(x) for x in re.findall(r"CoreTable\s+\S+:\s*(\d+)", stdout)]
    leaf_values = [int(x) for x in re.findall(r"LeafTable\s+\S+:\s*(\d+)", stdout)]
    table_values = core_values + leaf_values
    return {
        "internal_load_graph_seconds": first_float(r"Load graphs time \(seconds\):\s*([0-9.eE+-]+)"),
        "internal_filter_vertices_seconds": first_float(r"Filter vertices time \(seconds\):\s*([0-9.eE+-]+)"),
        "internal_build_table_seconds": first_float(r"Build table time \(seconds\):\s*([0-9.eE+-]+)"),
        "internal_generate_plan_seconds": first_float(r"Generate query plan time \(seconds\):\s*([0-9.eE+-]+)"),
        "internal_enumerate_seconds": first_float(r"Enumerate time \(seconds\):\s*([0-9.eE+-]+)"),
        "internal_preprocessing_seconds": first_float(r"Preprocessing time \(seconds\):\s*([0-9.eE+-]+)"),
        "internal_total_seconds": first_float(r"Total time \(seconds\):\s*([0-9.eE+-]+)"),
        "internal_total_cardinality": first_float(r"Total Cardinality:\s*([0-9.eE+-]+)"),
        "core_table_count": len(core_values),
        "core_table_sum": sum(core_values),
        "core_table_max": max(core_values) if core_values else 0,
        "leaf_table_count": len(leaf_values),
        "leaf_table_sum": sum(leaf_values),
        "leaf_table_max": max(leaf_values) if leaf_values else 0,
        "candidate_table_count": len(table_values),
        "candidate_table_sum": sum(table_values),
        "candidate_table_max": max(table_values) if table_values else 0,
    }


def run_solver(binary: Path, target_path: Path, query_path: Path, preset: Dict[str, str], timeout: float, max_solutions: int):
    cmd = [
        str(binary),
        "-d", str(target_path),
        "-q", str(query_path),
        "-filter", preset["filter"],
        "-order", preset["order"],
        "-engine", preset["engine"],
        "-num", str(max_solutions),
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        return {
            "returncode": proc.returncode,
            "timed_out": False,
            "seconds": elapsed,
            "embedding_count": parse_embedding_count(stdout),
            **parse_solver_metrics(stdout),
            "stdout_tail": stdout[-20000:],
            "stderr_tail": stderr[-1200:],
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "returncode": "",
            "timed_out": True,
            "seconds": elapsed,
            "embedding_count": parse_embedding_count(stdout),
            **parse_solver_metrics(stdout),
            "stdout_tail": stdout[-20000:],
            "stderr_tail": stderr[-1200:],
        }


def write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cora", choices=["cora", "arxiv", "mag"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--hierarchy-path", default="")
    parser.add_argument("--cache-dir", default="runs/subgraphmatching_baseline/cache")
    parser.add_argument("--output-dir", default="runs/subgraphmatching_baseline")
    parser.add_argument("--binary", default=os.environ.get("SUBGRAPH_MATCHING_BIN", "SubgraphMatching.out"))
    parser.add_argument("--query-types", default="k_hop,random_walk,multi_coarse")
    parser.add_argument("--target-sizes", default="20,50,100")
    parser.add_argument("--queries-per-cell", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--algorithms", default="cfl,dpiso,gql")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-solutions", default="1")
    parser.add_argument(
        "--label-source",
        default="feature",
        help=(
            "'feature' discretizes node features into integer labels, matching the "
            "legacy exact-solver/Jigsaw protocol; 'feature_bucket_K' applies "
            "md5(feature) mod K for label-selectivity stress tests; 'class' is "
            "a Cora diagnostic."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=0, help="Optional total solver-call cap for smoke tests.")
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    if not binary.exists():
        raise FileNotFoundError(f"SubgraphMatching binary not found: {binary}")

    output_dir = Path(args.output_dir).resolve()
    graph_dir = output_dir / "graphs"
    cache_dir = Path(args.cache_dir).resolve()

    data = cascade.load_named_data(args.dataset, args.data_root)
    labels = labels_for_source(data, args.label_source)
    data.node_label = labels
    label_counts = torch.bincount(labels.detach().cpu().view(-1).long())
    class_venue_base = None
    if args.label_source == "class" and hasattr(data, "y") and data.y is not None:
        y = data.y.detach().cpu().view(-1).long()
        class_venue_base = int(y[y >= 0].max().item()) + 1 if bool((y >= 0).any()) else 1

    hierarchy_path = args.hierarchy_path or cascade.default_hierarchy_path(args.dataset)
    cache_key = cascade.safe_cache_key(
        args.dataset,
        cascade.path_fingerprint(hierarchy_path),
        data.num_nodes,
        data.edge_index.size(1),
    )
    hierarchy = cascade.load_or_prepare_hierarchy(data, hierarchy_path, str(cache_dir), cache_key, args.dataset)
    hierarchy = cascade.load_or_build_overlap_index(data, hierarchy, str(cache_dir), cache_key)
    queries = cascade.load_or_generate_cascade_queries(
        data,
        hierarchy,
        args.queries_per_cell,
        args.target_sizes,
        args.seed,
        args.query_types,
        str(cache_dir),
        cache_key,
    )

    target_path = graph_dir / f"{args.dataset}_full.graph"
    target_meta = write_subgraphmatching_graph(data, target_path, labels)
    print(f"[LABELS] source={args.label_source} count={target_meta['label_count']}", flush=True)
    print(f"[TARGET] {target_path} {target_meta}", flush=True)

    rows = []
    algorithms = parse_csv(args.algorithms)
    sizes = set(parse_int_csv(args.target_sizes))
    max_runs = int(args.max_runs or 0)
    run_count = 0
    for query_index, item in enumerate(queries):
        query = item["query"]
        q_labels = torch.tensor(
            cascade.derive_query_labels(
                query,
                args.label_source,
                class_venue_base=class_venue_base,
            ),
            dtype=torch.long,
        )
        query.node_label = q_labels
        q_label_counts = label_counts[q_labels.long()].detach().cpu().long()
        label_consistent = bool(torch.equal(q_labels.cpu().long(), labels[item["query_nodes"].long()].cpu().long()))
        q_path = graph_dir / "queries" / f"{query_index:04d}_{item['query_type']}_n{item['target_query_size']}.graph"
        q_meta = write_subgraphmatching_graph(query, q_path, q_labels)
        for algorithm in algorithms:
            if algorithm not in PRESETS:
                raise ValueError(f"Unknown algorithm {algorithm}; choices={sorted(PRESETS)}")
            result = run_solver(
                binary,
                target_path,
                q_path,
                PRESETS[algorithm],
                args.timeout_seconds,
                args.max_solutions,
            )
            row = {
                "dataset": args.dataset,
                "algorithm": algorithm,
                "filter": PRESETS[algorithm]["filter"],
                "order": PRESETS[algorithm]["order"],
                "engine": PRESETS[algorithm]["engine"],
                "seed": args.seed,
                "query_index": query_index,
                "query_id": item["query_id"],
                "query_type": item["query_type"],
                "target_query_size": item["target_query_size"],
                "expected_match": bool(item.get("expected_match", True)),
                "query_nodes": q_meta["nodes"],
                "query_pruning_source": "query_payload_v1",
                "query_edges": q_meta["edges"],
                "target_nodes": target_meta["nodes"],
                "target_edges": target_meta["edges"],
                "label_source": args.label_source,
                "label_count": target_meta["label_count"],
                "label_consistent_with_planted_nodes": label_consistent,
                "query_unique_labels": int(q_labels.unique().numel()),
                "query_label_candidate_min": int(q_label_counts.min().item()) if int(q_label_counts.numel()) else 0,
                "query_label_candidate_mean": f"{float(q_label_counts.float().mean().item()):.6f}" if int(q_label_counts.numel()) else "0.000000",
                "query_label_candidate_max": int(q_label_counts.max().item()) if int(q_label_counts.numel()) else 0,
                "query_label_candidate_sum": int(q_label_counts.sum().item()) if int(q_label_counts.numel()) else 0,
                "timed_out": result["timed_out"],
                "returncode": result["returncode"],
                "seconds": f"{result['seconds']:.6f}",
                "embedding_count": result["embedding_count"],
                "found": int(result["embedding_count"]) > 0,
                "internal_load_graph_seconds": result["internal_load_graph_seconds"],
                "internal_filter_vertices_seconds": result["internal_filter_vertices_seconds"],
                "internal_build_table_seconds": result["internal_build_table_seconds"],
                "internal_generate_plan_seconds": result["internal_generate_plan_seconds"],
                "internal_enumerate_seconds": result["internal_enumerate_seconds"],
                "internal_preprocessing_seconds": result["internal_preprocessing_seconds"],
                "internal_total_seconds": result["internal_total_seconds"],
                "internal_total_cardinality": result["internal_total_cardinality"],
                "core_table_count": result["core_table_count"],
                "core_table_sum": result["core_table_sum"],
                "core_table_max": result["core_table_max"],
                "leaf_table_count": result["leaf_table_count"],
                "leaf_table_sum": result["leaf_table_sum"],
                "leaf_table_max": result["leaf_table_max"],
                "candidate_table_count": result["candidate_table_count"],
                "candidate_table_sum": result["candidate_table_sum"],
                "candidate_table_max": result["candidate_table_max"],
                "stdout_tail": result["stdout_tail"].replace("\n", "\\n"),
                "stderr_tail": result["stderr_tail"].replace("\n", "\\n"),
            }
            rows.append(row)
            print(
                f"[RUN] {algorithm} {item['query_id']} n={item['target_query_size']} "
                f"found={row['found']} timeout={row['timed_out']} seconds={row['seconds']}",
                flush=True,
            )
            run_count += 1
            if max_runs and run_count >= max_runs:
                break
        if max_runs and run_count >= max_runs:
            break

    out_csv = output_dir / f"{args.dataset}_subgraphmatching_probe.csv"
    write_rows(out_csv, rows)
    print(f"[DONE] wrote {len(rows)} rows to {out_csv}", flush=True)


if __name__ == "__main__":
    main()
