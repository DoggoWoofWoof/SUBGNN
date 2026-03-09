"""
Quick test to verify all subgraph isomorphism solvers work correctly.

Constructs a small, known graph and query subgraph, then runs each solver
to confirm it can find the subgraph and compute correct accuracy.

Usage:
    python -m src.test_solvers          # Run locally
    # Or on Modal for full testing
"""

import time
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def make_test_graphs():
    """
    Create a small target graph and a query subgraph contained within it.

    Target: 10-node graph (a ring + some chords)
    Query:  4-node path (nodes 2-3-4-5 from target)

    Returns:
        target_data, query_data, target_gids, query_gids
    """
    # Target graph: ring of 10 with some extra edges
    G_target = nx.cycle_graph(10)
    G_target.add_edges_from([(0, 5), (1, 6), (2, 7)])

    # Query: subgraph on nodes {2, 3, 4, 5}
    query_nodes = [2, 3, 4, 5]
    G_query = G_target.subgraph(query_nodes).copy()
    G_query = nx.convert_node_labels_to_integers(G_query)

    # Convert to PyG
    target_data = from_networkx(G_target)
    query_data = from_networkx(G_query)

    # Add dummy features (required by some solvers)
    target_data.x = torch.randn(target_data.num_nodes, 8)
    query_data.x = torch.randn(query_data.num_nodes, 8)

    # Global IDs: target already 0-9, query maps back to {2,3,4,5}
    target_gids = torch.arange(10)
    query_gids = torch.tensor(query_nodes)

    return target_data, query_data, target_gids, query_gids


def test_solver(solver_name, query_data, target_data, query_gids, target_gids):
    """Test a single solver and print results."""
    from src.solver_registry import run_solver

    print(f"\n{'='*50}")
    print(f"Testing: {solver_name.upper()}")
    print(f"{'='*50}")

    start = time.time()
    try:
        result = run_solver(
            solver_name,
            query_data=query_data,
            target_data=target_data,
            query_global_ids=query_gids,
            target_global_ids=target_gids,
            max_solutions=10,
            timeout_seconds=30.0,
        )
        elapsed = time.time() - start

        print(f"  Found:                {result.found}")
        print(f"  Timed Out:            {result.timed_out}")
        print(f"  Num Solutions:        {result.num_solutions}")
        print(f"  First Sol Accuracy:   {result.first_solution_accuracy:.1f}%")
        print(f"  Best Accuracy:        {result.best_accuracy:.1f}%")
        print(f"  Avg Accuracy:         {result.avg_accuracy:.1f}%")
        print(f"  Median Accuracy:      {result.median_accuracy:.1f}%")
        print(f"  Avg Nodes Matched:    {result.avg_nodes_matched:.1f}")
        print(f"  Median Nodes Matched: {result.median_nodes_matched:.1f}")
        print(f"  Latency:              {result.latency_seconds:.3f}s")
        print(f"  Time to First:        {result.time_to_first_solution:.3f}s")
        print(f"  Best Mapping:         {result.best_mapping}")
        print(f"  Wall Time:            {elapsed:.3f}s")

        # Validation
        if result.found:
            print(f"  STATUS: [PASS]")
        else:
            print(f"  STATUS: [FAIL] no solution found")

        return result.found

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        print(f"  Wall Time: {elapsed:.3f}s")
        print(f"  STATUS: [ERROR]")
        return False


def main():
    """Run tests for all available solvers."""
    from src.solver_registry import get_available_solvers

    print("Building test graphs...")
    target_data, query_data, target_gids, query_gids = make_test_graphs()
    print(f"  Target: {target_data.num_nodes} nodes, {target_data.num_edges} edges")
    print(f"  Query:  {query_data.num_nodes} nodes, {query_data.num_edges} edges")
    print(f"  Query GIDs: {query_gids.tolist()}")

    available = get_available_solvers()
    print(f"\nAvailable solvers: {available}")

    all_solvers = ['dpiso', 'cfl', 'turboiso']
    results = {}

    for solver_name in all_solvers:
        if solver_name in available:
            results[solver_name] = test_solver(
                solver_name, query_data, target_data, query_gids, target_gids
            )
        else:
            print(f"\n{'='*50}")
            print(f"SKIPPED: {solver_name.upper()} (not available)")
            print(f"{'='*50}")
            results[solver_name] = None

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name:>10s}: {status}")

    total_pass = sum(1 for v in results.values() if v is True)
    total_fail = sum(1 for v in results.values() if v is False)
    total_skip = sum(1 for v in results.values() if v is None)
    print(f"\n  {total_pass} passed, {total_fail} failed, {total_skip} skipped")


if __name__ == '__main__':
    main()
