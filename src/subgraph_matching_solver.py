"""
SubgraphMatching Solver Wrapper (DP-iso, CFL, TurboISO)

Wraps the RapidsAtHKUST/SubgraphMatching C++ binary.
Build: clone repo → cmake .. && make → SubgraphMatching.out

Graph format: vertex-labeled text file
    t # 0
    v <id> <label>
    e <src> <dst> <edge_label>

The binary outputs embeddings (mappings) to stdout.
"""

import os
import re
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.solver_registry import (
    MatchResult,
    compute_mapping_accuracy,
    aggregate_solution_metrics,
)


# ---------------------------------------------------------------------------
# Default binary path (set via env var or config)
# ---------------------------------------------------------------------------
_DEFAULT_BINARY = os.environ.get(
    'SUBGRAPH_MATCHING_BIN',
    '/app/SubgraphMatching/build/matching/SubgraphMatching.out',
)

# Algorithm presets — use recommended LFTJ engine per README
# The README recommends: GQL or CFL or DPiso for filtering,
# GQL or RI for ordering, and LFTJ for enumeration.
ALGORITHM_PRESETS = {
    'DPiso': {'filter': 'DPiso', 'order': 'DPiso', 'engine': 'DPiso'},
    'CFL':   {'filter': 'CFL',   'order': 'CFL',   'engine': 'LFTJ'},
    'TSO':   {'filter': 'TSO',   'order': 'TSO',    'engine': 'LFTJ'},
    'GQL':   {'filter': 'GQL',   'order': 'GQL',   'engine': 'LFTJ'},
}


def _pyg_to_graph_file(data: Data, filepath: str, label_map: Optional[Dict[int, int]] = None):
    """
    Write PyG data to SubgraphMatching binary format:

        t 0 N
        v VertexID LabelId Degree
        e VertexId VertexId

    If label_map is None, all nodes get label 0.
    Degree is computed from edge_index.
    """
    import networkx as nx
    from torch_geometric.utils import to_networkx

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # Build undirected edge set and compute degree
    edges_undirected = set()
    degree = [0] * num_nodes
    for col in range(edge_index.shape[1]):
        src = edge_index[0, col].item()
        dst = edge_index[1, col].item()
        if src == dst:
            continue
        edge_key = (min(src, dst), max(src, dst))
        edges_undirected.add(edge_key)

    # Compute degree from undirected edges
    for src, dst in edges_undirected:
        degree[src] += 1
        degree[dst] += 1

    num_edges = len(edges_undirected)

    with open(filepath, 'w') as f:
        f.write(f't {num_nodes} {num_edges}\n')

        # Vertices: v VertexID LabelId Degree
        for i in range(num_nodes):
            lbl = label_map.get(i, 0) if label_map else 0
            f.write(f'v {i} {lbl} {degree[i]}\n')

        # Edges: e src dst (no edge label)
        for src, dst in sorted(edges_undirected):
            f.write(f'e {src} {dst}\n')


def _parse_output(stdout_text: str, query_num_nodes: int) -> dict:
    """
    Parse output from SubgraphMatching.out.

    With our C++ patch, the binary prints:
        Embedding 1: v0 v1 v2 v3
        Embedding 2: v0 v1 v2 v3
        ...
        #Embeddings: N
        Enumerate time (seconds): 0.001

    Returns dict with embeddings list and timing info.
    """
    result = {
        'embeddings': [],       # List of {query_node: target_node} dicts
        'embedding_count': 0,   # From #Embeddings line (fallback)
        'enumeration_time': -1.0,
        'total_time': -1.0,
    }

    emb_pattern = re.compile(r'^Embedding\s+\d+:\s*(.+)$')

    for line in stdout_text.splitlines():
        line = line.strip()

        # Parse embedding lines: "Embedding N: v0 v1 v2 ..."
        m = emb_pattern.match(line)
        if m:
            values = m.group(1).strip().split()
            mapping = {}
            for q_node, t_node_str in enumerate(values):
                try:
                    mapping[q_node] = int(t_node_str)
                except ValueError:
                    pass
            if mapping:
                result['embeddings'].append(mapping)
            continue

        # Parse embedding count: "#Embeddings: N"
        if line.startswith('#Embeddings:') or line.startswith('Embedding Cnt:'):
            try:
                result['embedding_count'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        # Parse enumeration time
        elif 'numerat' in line.lower() and 'time' in line.lower():
            m = re.search(r'([\d.]+)', line.split(':')[-1])
            if m:
                result['enumeration_time'] = float(m.group(1))
        # Parse total time
        elif line.lower().startswith('total time'):
            m = re.search(r'([\d.]+)', line.split(':')[-1])
            if m:
                result['total_time'] = float(m.group(1))

    return result


def subgraph_matching_solve(
    query_data: Data,
    target_data: Data,
    query_global_ids: torch.Tensor,
    target_global_ids: torch.Tensor,
    max_solutions: int = 100,
    timeout_seconds: float = 300.0,
    algorithm: str = 'DPiso',
    binary_path: str = None,
    **kwargs,
) -> MatchResult:
    """
    Run subgraph matching using the RapidsAtHKUST binary.

    Args:
        algorithm: 'DPiso', 'CFL', or 'TSO' (TurboISO)
    """
    if binary_path is None:
        binary_path = _DEFAULT_BINARY

    if not os.path.exists(binary_path):
        print(f"[WARN] SubgraphMatching binary not found at {binary_path}")
        return MatchResult(
            found=False, timed_out=False,
            latency_seconds=0.0,
            solver_name=algorithm.lower(),
        )

    if algorithm not in ALGORITHM_PRESETS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {list(ALGORITHM_PRESETS)}")

    preset = ALGORITHM_PRESETS[algorithm]

    # Write temp graph files
    q_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.graph', delete=False, prefix='query_'
    )
    t_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.graph', delete=False, prefix='target_'
    )
    q_path = q_file.name
    t_path = t_file.name
    q_file.close()
    t_file.close()

    try:
        query_labels = kwargs.get('query_labels')
        target_labels = kwargs.get('target_labels')
        
        _pyg_to_graph_file(query_data, q_path, label_map=query_labels)
        _pyg_to_graph_file(target_data, t_path, label_map=target_labels)

        # Build command
        cmd = [
            binary_path,
            '-d', t_path,
            '-q', q_path,
            '-filter', preset['filter'],
            '-order', preset['order'],
            '-engine', preset['engine'],
            '-num', str(max_solutions),
        ]

        start_time = time.time()
        timed_out = False

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            stdout = proc.stdout
            stderr = proc.stderr

            # Check for process crashes (killed by signal)
            if proc.returncode < 0:
                import signal
                sig_num = -proc.returncode
                sig_names = {4: 'SIGILL', 6: 'SIGABRT', 11: 'SIGSEGV'}
                sig_name = sig_names.get(sig_num, f'signal {sig_num}')
                print(f"[WARN] SubgraphMatching binary crashed: {sig_name} (code {proc.returncode})", flush=True)
                if stderr:
                    print(f"[WARN] stderr: {stderr.strip()}", flush=True)
                if sig_num == 4:
                    print(f"[WARN] SIGILL = CPU does not support required instructions (AVX2?)", flush=True)
                return MatchResult(
                    found=False, timed_out=False,
                    latency_seconds=time.time() - start_time,
                    solver_name=algorithm.lower(),
                )
            elif proc.returncode != 0:
                print(f"[WARN] SubgraphMatching binary exited with code {proc.returncode}")
                print(f"[WARN] stderr: {stderr[:200]}")

        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout = e.stdout or ''
            stderr = e.stderr or ''
            if isinstance(stdout, bytes):
                stdout = stdout.decode('utf-8', errors='replace')
            if isinstance(stderr, bytes):
                stderr = stderr.decode('utf-8', errors='replace')

        latency = time.time() - start_time

        # Parse output — gets both embedding lines (if patched) and count/timing
        parsed = _parse_output(stdout, query_data.num_nodes)
        mappings = parsed['embeddings']
        embedding_count = len(mappings) if mappings else parsed['embedding_count']

        if mappings:
            # We have actual mappings — compute real accuracy
            accuracies = []
            nodes_matched_list = []
            best_idx = -1
            best_acc = -1.0

            for i, mapping in enumerate(mappings):
                acc, correct, total = compute_mapping_accuracy(
                    mapping, query_global_ids, target_global_ids
                )
                accuracies.append(acc)
                nodes_matched_list.append(correct)
                if acc > best_acc:
                    best_acc = acc
                    best_idx = i

            agg = aggregate_solution_metrics(accuracies, nodes_matched_list)
            best_mapping = mappings[best_idx] if 0 <= best_idx < len(mappings) else None
            time_to_first = parsed['enumeration_time'] if parsed['enumeration_time'] > 0 else latency / max(len(mappings), 1)

            return MatchResult(
                found=True,
                timed_out=timed_out,
                num_solutions=len(mappings),
                first_solution_accuracy=agg['first_solution_accuracy'],
                best_accuracy=agg['best_accuracy'],
                avg_accuracy=agg['avg_accuracy'],
                median_accuracy=agg['median_accuracy'],
                avg_nodes_matched=agg['avg_nodes_matched'],
                median_nodes_matched=agg['median_nodes_matched'],
                latency_seconds=latency,
                time_to_first_solution=time_to_first,
                best_mapping=best_mapping,
                solver_name=algorithm.lower(),
            )
        else:
            # No embedding lines — count-only (unpatched binary)
            return MatchResult(
                found=embedding_count > 0,
                timed_out=timed_out,
                num_solutions=embedding_count,
                first_solution_accuracy=-1.0,
                best_accuracy=-1.0,
                avg_accuracy=-1.0,
                median_accuracy=-1.0,
                avg_nodes_matched=-1.0,
                median_nodes_matched=-1.0,
                latency_seconds=latency,
                time_to_first_solution=parsed['enumeration_time'] if embedding_count > 0 else -1.0,
                best_mapping=None,
                solver_name=algorithm.lower(),
            )

    finally:
        try:
            os.unlink(q_path)
        except OSError:
            pass
        try:
            os.unlink(t_path)
        except OSError:
            pass


def make_solver(algorithm: str) -> Callable:
    """
    Create a solver function for a specific algorithm preset.

    Usage:
        dpiso_solve = make_solver('DPiso')
        result = dpiso_solve(query_data, target_data, ...)
    """
    def solver_fn(**kwargs):
        return subgraph_matching_solve(algorithm=algorithm, **kwargs)
    return solver_fn
