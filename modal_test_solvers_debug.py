"""
Debug Modal test — diagnose why solvers fail.
Runs direct vf3py + SubgraphMatching binary and prints raw output.

Usage:
    modal run modal_test_solvers_debug.py
"""

import os
import modal

app = modal.App("solver-debug")

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
    timeout=300,
    cpu=2,
    memory=4096,
    gpu="T4",  # T4 machines have AVX2-capable CPUs
)
def debug_solvers():
    """Debug solver issues."""
    import sys
    import subprocess
    import tempfile
    os.chdir("/app")
    sys.path.insert(0, "/app")

    import torch
    import networkx as nx
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx

    # --- Build test graphs ---
    G_target = nx.cycle_graph(10)
    G_target.add_edges_from([(0, 5), (1, 6), (2, 7)])
    query_nodes = [2, 3, 4, 5]
    G_query = G_target.subgraph(query_nodes).copy()
    G_query = nx.convert_node_labels_to_integers(G_query)

    print("=== Target graph ===")
    print(f"  Nodes: {list(G_target.nodes())}")
    print(f"  Edges: {list(G_target.edges())}")
    print(f"=== Query graph ===")
    print(f"  Nodes: {list(G_query.nodes())}")
    print(f"  Edges: {list(G_query.edges())}")

    # --- Test 1: Direct vf3py ---
    print("\n=== TEST 1: Direct vf3py ===")
    try:
        import vf3py
        print(f"vf3py imported, version: {vf3py.__version__ if hasattr(vf3py, '__version__') else '?'}")

        # Try different API calls
        print("\nTrying vf3py.get_subgraph_isomorphisms(G_target, G_query)...")
        try:
            results = list(vf3py.get_subgraph_isomorphisms(G_target, G_query))
            print(f"  Found {len(results)} isomorphisms")
            for i, m in enumerate(results[:3]):
                print(f"  Mapping {i}: {m}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print("\nTrying vf3py.get_subgraph_isomorphisms(G_query, G_target)...")
        try:
            results = list(vf3py.get_subgraph_isomorphisms(G_query, G_target))
            print(f"  Found {len(results)} isomorphisms")
            for i, m in enumerate(results[:3]):
                print(f"  Mapping {i}: {m}")
        except Exception as e:
            print(f"  ERROR: {e}")

        # Also try graph isomorphism
        print("\nTrying vf3py.get_graph_isomorphisms(G_query, G_query)...")
        try:
            results = list(vf3py.get_graph_isomorphisms(G_query, G_query))
            print(f"  Found {len(results)} isomorphisms")
            for i, m in enumerate(results[:3]):
                print(f"  Mapping {i}: {m}")
        except Exception as e:
            print(f"  ERROR: {e}")

        # Check what functions are available
        print(f"\nvf3py dir: {[x for x in dir(vf3py) if not x.startswith('_')]}")

    except ImportError as e:
        print(f"vf3py import error: {e}")

    # --- Test 2: NetworkX GraphMatcher ---
    print("\n=== TEST 2: NetworkX GraphMatcher ===")
    from networkx.algorithms.isomorphism import GraphMatcher
    GM = GraphMatcher(G_target, G_query)
    results = list(GM.subgraph_isomorphisms_iter())
    print(f"  Found {len(results)} subgraph isomorphisms")
    for i, m in enumerate(results[:3]):
        print(f"  Mapping {i}: {m}")

    # --- Test 3: SubgraphMatching binary ---
    print("\n=== TEST 3: SubgraphMatching binary ===")
    binary = "/app/SubgraphMatching/build/matching/SubgraphMatching.out"
    print(f"  Binary exists: {os.path.exists(binary)}")

    # Check CPU flags
    try:
        with open('/proc/cpuinfo') as f:
            cpuinfo = f.read()
        has_avx2 = 'avx2' in cpuinfo
        has_sse42 = 'sse4_2' in cpuinfo
        print(f"  CPU has AVX2: {has_avx2}")
        print(f"  CPU has SSE4.2: {has_sse42}")
    except:
        print("  Could not read /proc/cpuinfo")

    # Check ldd
    try:
        ldd_result = subprocess.run(['ldd', binary], capture_output=True, text=True, timeout=5)
        print(f"  ldd output:\n{ldd_result.stdout[:500]}")
    except:
        print("  ldd failed")

    # Try running binary without args
    print("\n  --- Running binary with no args ---")
    try:
        proc = subprocess.run([binary], capture_output=True, text=True, timeout=5)
        print(f"  Return code: {proc.returncode}")
        print(f"  STDOUT: {proc.stdout[:300]}")
        print(f"  STDERR: {proc.stderr[:300]}")
    except Exception as e:
        print(f"  Error: {e}")

    # Write graph files in correct format: t N M, v ID Label Degree, e src dst
    q_path = "/tmp/test_query.graph"
    t_path = "/tmp/test_target.graph"

    # Target
    with open(t_path, 'w') as f:
        n_nodes = G_target.number_of_nodes()
        n_edges = G_target.number_of_edges()
        f.write(f"t {n_nodes} {n_edges}\n")
        for n in G_target.nodes():
            f.write(f"v {n} 0 {G_target.degree(n)}\n")
        for u, v in G_target.edges():
            f.write(f"e {u} {v}\n")

    # Query
    with open(q_path, 'w') as f:
        n_nodes = G_query.number_of_nodes()
        n_edges = G_query.number_of_edges()
        f.write(f"t {n_nodes} {n_edges}\n")
        for n in G_query.nodes():
            f.write(f"v {n} 0 {G_query.degree(n)}\n")
        for u, v in G_query.edges():
            f.write(f"e {u} {v}\n")

    print(f"\n  Target file:")
    with open(t_path) as f:
        print(f.read())
    print(f"  Query file:")
    with open(q_path) as f:
        print(f.read())

    # Run binary with different algorithms
    for algo_filter, algo_order, algo_engine, name in [
        ("DPiso", "DPiso", "DPiso", "DP-iso"),
        ("CFL", "CFL", "LFTJ", "CFL"),
        ("TSO", "TSO", "LFTJ", "TurboISO"),
        ("GQL", "GQL", "LFTJ", "GQL"),
    ]:
        print(f"\n  --- {name} ---")
        cmd = [
            binary, "-d", t_path, "-q", q_path,
            "-filter", algo_filter, "-order", algo_order, "-engine", algo_engine,
            "-num", "10",
        ]
        print(f"  CMD: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"  Return code: {proc.returncode}")
        print(f"  STDOUT:\n{proc.stdout}")
        if proc.stderr:
            print(f"  STDERR:\n{proc.stderr}")

    return "Done"


@app.local_entrypoint()
def main():
    result = debug_solvers.remote()
    print(result)
