"""
Unified Solver Registry

Provides a common MatchResult dataclass and dispatch function for all
subgraph isomorphism solvers (VF3, DP-iso, CFL, TurboISO).
"""

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
from torch_geometric.data import Data


@dataclass
class MatchResult:
    """Unified result from any subgraph isomorphism solver."""

    # Core result
    found: bool = False                          # At least one valid mapping found
    timed_out: bool = False                      # Hit the timeout
    num_solutions: int = 0                       # Solutions found (capped at max_solutions)

    # Accuracy (% of query nodes correctly mapped to ground-truth global IDs)
    first_solution_accuracy: float = -1.0        # Accuracy of the 1st solution (-1 if none)
    best_accuracy: float = -1.0                  # Max accuracy across all solutions
    avg_accuracy: float = -1.0                   # Mean accuracy across solutions
    median_accuracy: float = -1.0                # Median accuracy across solutions

    # Node-level
    avg_nodes_matched: float = -1.0              # Avg correctly-mapped nodes per solution
    median_nodes_matched: float = -1.0           # Median correctly-mapped nodes per solution

    # Timing
    latency_seconds: float = 0.0                 # Total wall time
    time_to_first_solution: float = -1.0         # Time to first solution (-1 if none)

    # Best mapping (query_local -> target_local)
    best_mapping: Optional[Dict[int, int]] = None

    # Solver metadata
    solver_name: str = ""


def compute_mapping_accuracy(
    mapping: Dict[int, int],
    query_global_ids: torch.Tensor,
    target_global_ids: torch.Tensor,
) -> tuple:
    """
    Compute accuracy of a node mapping against ground truth.

    A correct mapping means: query_global_ids[q_local] == target_global_ids[t_local]

    Returns:
        (accuracy_pct, num_correct, total_nodes)
    """
    if not mapping:
        return 0.0, 0, 0

    correct = 0
    total = len(mapping)

    for q_local, t_local in mapping.items():
        try:
            q_val = query_global_ids[q_local]
            t_val = target_global_ids[t_local]
            q_global = q_val.item() if hasattr(q_val, 'item') else int(q_val)
            t_global = t_val.item() if hasattr(t_val, 'item') else int(t_val)
            if q_global == t_global:
                correct += 1
        except (IndexError, KeyError):
            pass

    accuracy = (correct / total * 100) if total > 0 else 0.0
    return accuracy, correct, total


def aggregate_solution_metrics(
    accuracies: List[float],
    nodes_matched_list: List[int],
) -> dict:
    """Compute aggregate stats from per-solution metrics."""
    result = {}
    if accuracies:
        result['first_solution_accuracy'] = accuracies[0]
        result['best_accuracy'] = max(accuracies)
        result['avg_accuracy'] = statistics.mean(accuracies)
        result['median_accuracy'] = statistics.median(accuracies)
    else:
        result['first_solution_accuracy'] = -1.0
        result['best_accuracy'] = -1.0
        result['avg_accuracy'] = -1.0
        result['median_accuracy'] = -1.0

    if nodes_matched_list:
        result['avg_nodes_matched'] = statistics.mean(nodes_matched_list)
        result['median_nodes_matched'] = statistics.median(nodes_matched_list)
    else:
        result['avg_nodes_matched'] = -1.0
        result['median_nodes_matched'] = -1.0

    return result


# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------

_SOLVER_CLASSES: Dict[str, Callable] = {}


def register_solver(name: str, factory: Callable):
    """Register a solver factory function."""
    _SOLVER_CLASSES[name] = factory


def get_available_solvers() -> List[str]:
    """Return names of all registered solvers."""
    return list(_SOLVER_CLASSES.keys())


def run_solver(
    name: str,
    query_data: Data,
    target_data: Data,
    query_global_ids: torch.Tensor,
    target_global_ids: torch.Tensor,
    max_solutions: int = 100,
    timeout_seconds: float = 300.0,
    **kwargs,
) -> MatchResult:
    """
    Dispatch to the named solver.

    Args:
        name: One of 'vf3', 'dpiso', 'cfl', 'turboiso'
        query_data: PyG Data for query graph
        target_data: PyG Data for target graph
        query_global_ids: Global node IDs for query
        target_global_ids: Global node IDs for target
        max_solutions: Maximum solutions to enumerate
        timeout_seconds: Per-call timeout

    Returns:
        MatchResult with unified metrics
    """
    if name not in _SOLVER_CLASSES:
        raise ValueError(
            f"Unknown solver '{name}'. Available: {get_available_solvers()}"
        )

    solver_fn = _SOLVER_CLASSES[name]
    result = solver_fn(
        query_data=query_data,
        target_data=target_data,
        query_global_ids=query_global_ids,
        target_global_ids=target_global_ids,
        max_solutions=max_solutions,
        timeout_seconds=timeout_seconds,
        **kwargs,
    )
    result.solver_name = name
    return result


# ---------------------------------------------------------------------------
# Auto-register solvers on import
# ---------------------------------------------------------------------------

def _register_all():
    """Register all available solvers."""
    try:
        from src.subgraph_matching_solver import make_solver

        register_solver('dpiso', make_solver('DPiso'))
        register_solver('cfl', make_solver('CFL'))
        register_solver('turboiso', make_solver('TSO'))
    except ImportError:
        pass


_register_all()
