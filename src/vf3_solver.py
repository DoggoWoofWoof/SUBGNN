"""
VF3 Subgraph Isomorphism Solver

Uses vf3py (pip install vf3py) for VF3 algorithm.
Linux-only — designed to run on Modal.
"""

import time
from typing import Dict, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.solver_registry import (
    MatchResult,
    compute_mapping_accuracy,
    aggregate_solution_metrics,
)


def _pyg_to_networkx(data: Data):
    """Convert PyG Data to undirected NetworkX graph."""
    import networkx as nx

    G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    return G


def vf3_solve(
    query_data: Data,
    target_data: Data,
    query_global_ids: torch.Tensor,
    target_global_ids: torch.Tensor,
    max_solutions: int = 100,
    timeout_seconds: float = 300.0,
    **kwargs,
) -> MatchResult:
    """
    Run VF3 subgraph isomorphism via vf3py.

    Uses multiprocessing for hard timeout enforcement.
    Returns partial results if timed out.
    """
    from multiprocessing import Process, Queue

    result_queue = Queue()
    start_time = time.time()

    proc = Process(
        target=_vf3_worker,
        args=(
            query_data, target_data,
            query_global_ids, target_global_ids,
            max_solutions, timeout_seconds,
            result_queue, start_time,
        ),
    )
    proc.start()
    proc.join(timeout=timeout_seconds + 5)  # extra grace

    timed_out = False
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
        timed_out = True

    latency = time.time() - start_time

    # Collect result from queue
    try:
        if not result_queue.empty():
            worker_result = result_queue.get_nowait()
        else:
            worker_result = None
    except Exception:
        worker_result = None

    if worker_result is None:
        return MatchResult(
            found=False, timed_out=timed_out,
            latency_seconds=latency, solver_name='vf3',
        )

    # Unpack worker result
    accuracies = worker_result.get('accuracies', [])
    nodes_matched = worker_result.get('nodes_matched', [])
    best_idx = worker_result.get('best_idx', -1)
    time_to_first = worker_result.get('time_to_first', -1.0)
    timed_out = timed_out or worker_result.get('timed_out', False)
    mappings = worker_result.get('mappings', [])

    agg = aggregate_solution_metrics(accuracies, nodes_matched)

    best_mapping = mappings[best_idx] if 0 <= best_idx < len(mappings) else None

    return MatchResult(
        found=len(accuracies) > 0,
        timed_out=timed_out,
        num_solutions=len(accuracies),
        first_solution_accuracy=agg['first_solution_accuracy'],
        best_accuracy=agg['best_accuracy'],
        avg_accuracy=agg['avg_accuracy'],
        median_accuracy=agg['median_accuracy'],
        avg_nodes_matched=agg['avg_nodes_matched'],
        median_nodes_matched=agg['median_nodes_matched'],
        latency_seconds=latency,
        time_to_first_solution=time_to_first,
        best_mapping=best_mapping,
        solver_name='vf3',
    )


def _vf3_worker(
    query_data, target_data,
    query_global_ids, target_global_ids,
    max_solutions, timeout_seconds,
    result_queue, start_time,
):
    """Worker process for VF3 — pickleable at module level."""
    import time

    try:
        import networkx as nx
        from networkx.algorithms.isomorphism import GraphMatcher

        # Convert to NetworkX
        query_nx = _pyg_to_networkx(query_data)
        target_nx = _pyg_to_networkx(target_data)

        # Try vf3py first, fallback to NetworkX GraphMatcher
        iso_iter = None
        use_vf3py = False
        try:
            import vf3py
            # vf3py expects (query, target) — returns {query_node: target_node}
            iso_iter = vf3py.get_subgraph_isomorphisms(query_nx, target_nx)
            use_vf3py = True
        except (ImportError, Exception):
            pass

        if iso_iter is None:
            # Fallback: NetworkX GraphMatcher(G1=target, G2=query)
            # returns {target_node: query_node}
            GM = GraphMatcher(target_nx, query_nx)
            iso_iter = GM.subgraph_isomorphisms_iter()

        accuracies = []
        nodes_matched = []
        mappings = []
        best_idx = -1
        best_acc = -1.0
        time_to_first = -1.0
        timed_out = False

        for mapping_raw in iso_iter:
            now = time.time()
            elapsed = now - start_time

            if elapsed > timeout_seconds:
                timed_out = True
                break

            # vf3py returns {query_node: target_node} — use directly
            # GraphMatcher returns {target_node: query_node} — need inversion
            if use_vf3py:
                mapping = dict(mapping_raw)
            else:
                mapping = {v: k for k, v in mapping_raw.items()}

            if time_to_first < 0:
                time_to_first = elapsed

            # Compute accuracy
            acc, correct, total = compute_mapping_accuracy(
                mapping, query_global_ids, target_global_ids
            )

            accuracies.append(acc)
            nodes_matched.append(correct)
            mappings.append(mapping)

            if acc > best_acc:
                best_acc = acc
                best_idx = len(accuracies) - 1

            if len(accuracies) >= max_solutions:
                break

        result_queue.put({
            'accuracies': accuracies,
            'nodes_matched': nodes_matched,
            'mappings': mappings,
            'best_idx': best_idx,
            'time_to_first': time_to_first,
            'timed_out': timed_out,
        })

    except Exception as e:
        result_queue.put({
            'accuracies': [], 'nodes_matched': [], 'mappings': [],
            'best_idx': -1, 'time_to_first': -1.0, 'timed_out': False,
            'error': str(e),
        })
