import os
import subprocess
import tempfile
import time
import hashlib
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import torch
import numpy as np
from torch_geometric.data import Data

from src.utils import feature_to_label


# Global cache for target graphs to avoid redundant serialization
_target_lad_cache = {}


@dataclass
class GlasgowResult:
    """Structured result from Glasgow solver."""
    found: bool = False
    timed_out: bool = False
    num_solutions: int = 0
    latency_seconds: float = 0.0
    time_to_first_solution: float = -1.0
    first_solution_accuracy: float = -1.0
    best_accuracy: float = -1.0
    best_mapping: Dict[int, int] = field(default_factory=dict)
    all_mappings: List[Dict[int, int]] = field(default_factory=list)


def _pyg_to_vertexlabelledlad(data: Data, filepath: str):
    """
    Write a PyG graph to Glasgow's vertexlabelledlad format.
    
    Format:
        <num_vertices>
        <label_0> <degree_0> <neighbor_0_0> <neighbor_0_1> ...
        <label_1> <degree_1> <neighbor_1_0> <neighbor_1_1> ...
        ...
    
    Each line i: label_i degree_i adj_0 adj_1 ...
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # Build adjacency lists
    adj = [[] for _ in range(num_nodes)]
    if edge_index is not None and edge_index.shape[1] > 0:
        src_arr = edge_index[0].tolist()
        dst_arr = edge_index[1].tolist()
        for s, d in zip(src_arr, dst_arr):
            if s != d:  # skip self-loops
                adj[s].append(d)

    # Deduplicate adjacency lists
    for i in range(num_nodes):
        adj[i] = sorted(set(adj[i]))

    # Compute labels. Prefer an explicit per-node label (e.g. class label data.y)
    # when attached as data.node_label; otherwise fall back to the feature hash.
    labels = []
    node_label = getattr(data, "node_label", None)
    for i in range(num_nodes):
        if node_label is not None:
            label = int(node_label[i])
        elif hasattr(data, 'x') and data.x is not None:
            label = feature_to_label(data.x[i])
        else:
            label = 0
        labels.append(label)

    with open(filepath, 'w') as f:
        f.write(f"{num_nodes}\n")
        for i in range(num_nodes):
            neighbors = adj[i]
            line = f"{labels[i]} {len(neighbors)}"
            if neighbors:
                line += " " + " ".join(str(n) for n in neighbors)
            f.write(line + "\n")


def _compute_mapping_accuracy(mapping: Dict[int, int],
                               query_global_ids: torch.Tensor,
                               target_global_ids: torch.Tensor):
    """Compute accuracy: fraction of mapped nodes matching their original global IDs."""
    if not mapping:
        return 0.0, 0, 0
    correct = 0
    total = len(mapping)
    for q_idx, t_idx in mapping.items():
        if q_idx < len(query_global_ids) and t_idx < len(target_global_ids):
            if query_global_ids[q_idx] == target_global_ids[t_idx]:
                correct += 1
    return correct / total if total > 0 else 0.0, correct, total


def glasgow_solve(
    query_data: Data,
    target_data: Data,
    query_global_ids: torch.Tensor = None,
    target_global_ids: torch.Tensor = None,
    max_solutions: int = 1,
    timeout_seconds: float = 60.0,
    is_debug: bool = False,
    binary_path: str = "/usr/local/bin/glasgow_subgraph_solver",
    target_name: Optional[str] = None,
) -> GlasgowResult:
    """
    Solve subgraph isomorphism using the Glasgow Subgraph Solver.
    Uses vertexlabelledlad format for label-aware matching.
    
    Args:
        query_data: Query graph (PyG Data with .x features and .global_id)
        target_data: Target graph (PyG Data)
        query_global_ids: Global node IDs for the query (for accuracy)
        target_global_ids: Global node IDs for the target (for accuracy)
        max_solutions: Max solutions to collect
        timeout_seconds: Solver timeout
        binary_path: Path to glasgow_subgraph_solver binary
        target_name: Cache key for target graph serialization
    """
    result = GlasgowResult()

    if query_global_ids is None:
        query_global_ids = query_data.global_id if hasattr(query_data, 'global_id') else None
    if target_global_ids is None:
        target_global_ids = target_data.global_id if hasattr(target_data, 'global_id') else None

    # Sanity check
    if query_data.num_nodes > target_data.num_nodes:
        return result

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write pattern (query) file
        pattern_path = os.path.join(tmpdir, "pattern.lad")
        _pyg_to_vertexlabelledlad(query_data, pattern_path)

        # Write target file (with caching for repeated calls)
        if target_name and target_name in _target_lad_cache:
            target_path = _target_lad_cache[target_name]
        else:
            if target_name:
                target_path = os.path.join(tempfile.gettempdir(), f"glasgow_{target_name}.lad")
                if not os.path.exists(target_path):
                    _pyg_to_vertexlabelledlad(target_data, target_path)
                _target_lad_cache[target_name] = target_path
            else:
                target_path = os.path.join(tmpdir, "target.lad")
                _pyg_to_vertexlabelledlad(target_data, target_path)

        # Glasgow command: positional args are PATTERN TARGET
        cmd = [
            binary_path,
            "--format", "vertexlabelledlad",
            "--timeout", str(int(timeout_seconds)),
            "--print-all-solutions",
            pattern_path,
            target_path,
        ]

        if is_debug:
            print(f"[GLASGOW] cmd: {' '.join(cmd)}", flush=True)
            # Print first few lines of pattern file for debugging
            with open(pattern_path) as f:
                lines = f.readlines()[:5]
                print(f"[GLASGOW] pattern ({len(lines)} lines shown): {lines}", flush=True)

        start = time.time()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )

            # Watchdog: kill if solver exceeds timeout + 15s buffer
            import threading
            watchdog_killed = False

            def _watchdog():
                nonlocal watchdog_killed
                if proc.poll() is None:
                    try:
                        watchdog_killed = True
                        proc.kill()
                    except Exception:
                        pass
            watchdog = threading.Timer(timeout_seconds + 15, _watchdog)
            watchdog.daemon = True
            watchdog.start()

            # Stream stdout for solutions
            mappings = []
            status = "unknown"
            first_solution_time = None

            if proc.stdout:
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue

                    if is_debug:
                        print(f"[GLASGOW OUT] {line}", flush=True)

                    if line.startswith("mapping ="):
                        mapping_str = line.split("=", 1)[1].strip()
                        m_dict = {}
                        pairs = re.findall(r"\((\d+) -> (\d+)\)", mapping_str)
                        for q_node, t_node in pairs:
                            m_dict[int(q_node)] = int(t_node)
                        mappings.append(m_dict)

                        if first_solution_time is None:
                            first_solution_time = time.time() - start

                        if len(mappings) >= max_solutions:
                            break

                    elif line.startswith("status ="):
                        status = line.split("=", 1)[1].strip()

            # Cancel watchdog and clean up
            watchdog.cancel()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            
            # Check stderr
            stderr = proc.stderr.read() if proc.stderr else ""
            if stderr.strip() and is_debug:
                print(f"[GLASGOW STDERR] {stderr}", flush=True)

            latency = time.time() - start
            result.latency_seconds = latency
            result.timed_out = (
                status == "aborted"
                or watchdog_killed
                or latency >= max(float(timeout_seconds) - 0.5, 0.0)
            )

            if mappings:
                result.found = True
                result.num_solutions = len(mappings)
                result.all_mappings = mappings
                result.time_to_first_solution = first_solution_time or latency

                # Compute accuracies
                if query_global_ids is not None and target_global_ids is not None:
                    best_acc = -1.0
                    best_idx = 0
                    for i, m in enumerate(mappings):
                        acc, _, _ = _compute_mapping_accuracy(m, query_global_ids, target_global_ids)
                        if acc > best_acc:
                            best_acc = acc
                            best_idx = i

                    first_acc, _, _ = _compute_mapping_accuracy(mappings[0], query_global_ids, target_global_ids)
                    result.first_solution_accuracy = first_acc
                    result.best_accuracy = best_acc
                    result.best_mapping = mappings[best_idx]
                else:
                    result.first_solution_accuracy = 1.0
                    result.best_accuracy = 1.0
                    result.best_mapping = mappings[0]

            return result

        except Exception as e:
            if is_debug:
                print(f"[GLASGOW EXCEPTION] {str(e)}", flush=True)
            result.latency_seconds = time.time() - start
            return result
