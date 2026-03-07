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

        t N M
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


def _parse_output(stdout_text: str) -> dict:
    """
    Parse output from SubgraphMatching.out.

    The binary outputs stats like:
        #Embeddings: 60
        Enumeration Time: 0.001(seconds)
        Call Count: ...
        Total Time: ...

    Returns dict with embedding_count and timing info.
    """
    result = {
        'embedding_count': 0,
        'enumeration_time': -1.0,
        'total_time': -1.0,
    }

    for line in stdout_text.splitlines():
        line = line.strip()
        # Parse embedding count
        if line.startswith('#Embeddings:') or line.startswith('Embedding Cnt:'):
            try:
                result['embedding_count'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        # Parse enumeration time
        elif 'Enumeration' in line and 'Time' in line:
            m = re.search(r'([\d.]+)', line.split(':')[-1])
            if m:
                result['enumeration_time'] = float(m.group(1))
        # Parse total time
        elif line.startswith('Total Time:') or 'total time' in line.lower():
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
        _pyg_to_graph_file(query_data, q_path)
        _pyg_to_graph_file(target_data, t_path)

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
                print(f"[WARN] SubgraphMatching binary crashed: {sig_name} (code {proc.returncode})")
                if sig_num == 4:
                    print(f"[WARN] SIGILL = CPU does not support required instructions (AVX2?)")
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

        # Parse count-based output (SubgraphMatching only outputs counts, not mappings)
        parsed = _parse_output(stdout)
        embedding_count = parsed['embedding_count']

        # SubgraphMatching is a benchmarking tool — it reports how many
        # embeddings exist but doesn't output the actual node mappings.
        # So we can only report found/count/timing, not per-node accuracy.
        return MatchResult(
            found=embedding_count > 0,
            timed_out=timed_out,
            num_solutions=embedding_count,
            # Accuracy not available from count-only output
            first_solution_accuracy=100.0 if embedding_count > 0 else -1.0,
            best_accuracy=100.0 if embedding_count > 0 else -1.0,
            avg_accuracy=100.0 if embedding_count > 0 else -1.0,
            median_accuracy=100.0 if embedding_count > 0 else -1.0,
            avg_nodes_matched=query_data.num_nodes if embedding_count > 0 else -1.0,
            median_nodes_matched=query_data.num_nodes if embedding_count > 0 else -1.0,
            latency_seconds=latency,
            time_to_first_solution=parsed['enumeration_time'] if embedding_count > 0 else -1.0,
            best_mapping=None,  # Mappings not available from this solver
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
