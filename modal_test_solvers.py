"""
Modal-based test for all 4 subgraph isomorphism solvers.

Runs on Modal (Linux) to test VF3, DP-iso, CFL, and TurboISO.
This ensures vf3py works natively and SubgraphMatching.out binary is available.

Usage:
    modal run modal_test_solvers.py
"""

import os
import modal

app = modal.App("solver-test")

# Same image as modal_evaluate.py — with vf3py + SubgraphMatching build
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential")
    .pip_install(
        "torch==2.1.2",
        "numpy<2.0.0",
    )
    .pip_install(
        "torch-geometric",
        "torch-sparse",
        "torch-scatter",
        "torch-cluster",
        "scipy",
        "networkx",
        "faiss-cpu>=1.7.4",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "vf3py",
    )
    .run_commands(
        "git clone https://github.com/RapidsAtHKUST/SubgraphMatching.git /app/SubgraphMatching",
        "sed -i 's/#define SI 0/#define SI 2/' /app/SubgraphMatching/configuration/config.h",
        "sed -i 's/#define HYBRID 0/#define HYBRID 1/' /app/SubgraphMatching/configuration/config.h",
        "sed -i 's|// #define ENABLE_FAILING_SET|#define ENABLE_FAILING_SET|' /app/SubgraphMatching/configuration/config.h",
        "sed -i 's/-march=native/-march=x86-64-v2/' /app/SubgraphMatching/CMakeLists.txt",
        "cd /app/SubgraphMatching && mkdir -p build && cd build && cmake .. && make -j$(nproc)",
    )
)


@app.function(
    image=image.add_local_dir("src", remote_path="/app/src"),
    timeout=600,
    cpu=2,
    memory=4096,
)
def test_all_solvers():
    """
    Test all 4 solvers on Modal with a small known graph.
    Returns a dict of {solver_name: passed_bool}.
    """
    import time
    import sys
    os.chdir("/app")
    sys.path.insert(0, "/app")

    import torch
    import networkx as nx
    from torch_geometric.utils import from_networkx

    # --- Build test graphs ---
    # Target: 10-node ring with extra edges
    G_target = nx.cycle_graph(10)
    G_target.add_edges_from([(0, 5), (1, 6), (2, 7)])

    # Query: subgraph on nodes {2, 3, 4, 5}
    query_nodes = [2, 3, 4, 5]
    G_query = G_target.subgraph(query_nodes).copy()
    G_query = nx.convert_node_labels_to_integers(G_query)

    target_data = from_networkx(G_target)
    query_data = from_networkx(G_query)
    target_data.x = torch.randn(target_data.num_nodes, 8)
    query_data.x = torch.randn(query_data.num_nodes, 8)

    target_gids = torch.arange(10)
    query_gids = torch.tensor(query_nodes)

    print(f"Target: {target_data.num_nodes} nodes, {target_data.num_edges} edges")
    print(f"Query:  {query_data.num_nodes} nodes, {query_data.num_edges} edges")
    print(f"Query GIDs: {query_gids.tolist()}")

    # --- Check SubgraphMatching binary ---
    binary_path = "/app/SubgraphMatching/build/matching/SubgraphMatching.out"
    print(f"\nSubgraphMatching binary exists: {os.path.exists(binary_path)}")

    # --- Check vf3py ---
    try:
        import vf3py
        print(f"vf3py version: {vf3py.__version__ if hasattr(vf3py, '__version__') else 'installed'}")
    except ImportError:
        print("vf3py: NOT INSTALLED")

    # --- Import solver registry ---
    from src.solver_registry import run_solver, get_available_solvers

    available = get_available_solvers()
    print(f"\nRegistered solvers: {available}")

    # --- Run each solver ---
    all_solvers = ['vf3', 'dpiso', 'cfl', 'turboiso']
    results = {}

    for name in all_solvers:
        print(f"\n{'='*50}")
        print(f"Testing: {name.upper()}")
        print(f"{'='*50}")

        if name not in available:
            print(f"  SKIPPED (not registered)")
            results[name] = None
            continue

        start = time.time()
        try:
            result = run_solver(
                name,
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

            if result.found:
                print(f"  STATUS: [PASS]")
                results[name] = True
            else:
                print(f"  STATUS: [FAIL] no solution found")
                results[name] = False

        except Exception as e:
            elapsed = time.time() - start
            print(f"  ERROR: {e}")
            print(f"  Wall Time: {elapsed:.3f}s")
            print(f"  STATUS: [ERROR]")
            results[name] = False

    # --- Summary ---
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

    return results


@app.local_entrypoint()
def main():
    """Run solver tests on Modal."""
    print("Running solver tests on Modal...")
    results = test_all_solvers.remote()
    print(f"\n=== FINAL RESULTS: {results} ===")

