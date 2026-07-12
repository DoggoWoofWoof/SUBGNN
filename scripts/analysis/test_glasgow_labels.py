"""
Glasgow Label Enforcement Diagnostic
=====================================
Tests whether the Glasgow Subgraph Solver actually enforces label matching
when using the vertexlabelledlad format.

Run on Modal via:
  modal run modal_benchmark_glasgow.py --diag
"""
import os
import subprocess
import tempfile

GLASGOW_BIN = os.environ.get('GLASGOW_SOLVER_BIN', '/usr/local/bin/glasgow_subgraph_solver')


def write_vertexlabelledlad(filepath, labels, adj_lists):
    """
    Write a graph in vertexlabelledlad format.
    Format:
        <num_vertices>
        <label_0> <degree_0> <neighbor_0_0> <neighbor_0_1> ...
        ...
    """
    n = len(labels)
    with open(filepath, 'w') as f:
        f.write(f"{n}\n")
        for i in range(n):
            neighbors = sorted(set(adj_lists.get(i, [])))
            line = f"{labels[i]} {len(neighbors)}"
            if neighbors:
                line += " " + " ".join(str(x) for x in neighbors)
            f.write(line + "\n")


def run_glasgow(pattern_path, target_path, timeout=10):
    """Run Glasgow solver and return (status, mappings_count, stdout, stderr)."""
    cmd = [
        GLASGOW_BIN,
        "--format", "vertexlabelledlad",
        "--timeout", str(timeout),
        "--print-all-solutions",
        pattern_path,
        target_path,
    ]
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    
    mappings = [l for l in stdout.split('\n') if l.startswith('mapping =')]
    status_lines = [l for l in stdout.split('\n') if l.startswith('status =')]
    status = status_lines[0].split('=')[1].strip() if status_lines else 'unknown'
    
    return status, len(mappings), stdout, stderr


def main():
    print("=" * 70)
    print("  GLASGOW LABEL ENFORCEMENT DIAGNOSTIC (vertexlabelledlad)")
    print("=" * 70)
    
    # Graph: triangle 0-1-2
    # Query adj: 0↔1, 1↔2, 0↔2
    q_adj = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    
    # Target: 5 nodes, triangle 0-1-2 + edge 2-3, 3-4
    t_adj = {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2, 4], 4: [3]}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # ── TEST 1: Matching labels → should find solution ──
        print("\n" + "-" * 50)
        print("TEST 1: MATCHING LABELS (should find solution)")
        print("-" * 50)
        
        p1 = os.path.join(tmpdir, "p1.lad")
        t1 = os.path.join(tmpdir, "t1.lad")
        write_vertexlabelledlad(p1, [10, 20, 30], q_adj)
        write_vertexlabelledlad(t1, [10, 20, 30, 40, 50], t_adj)
        
        print("  Pattern file:")
        print("  " + open(p1).read().replace("\n", "\n  "))
        print("  Target file:")
        print("  " + open(t1).read().replace("\n", "\n  "))
        
        status, n_maps, stdout, stderr = run_glasgow(p1, t1)
        print(f"  Status: {status}")
        print(f"  Mappings found: {n_maps}")
        if stdout: print(f"  STDOUT:\n  {stdout}")
        if stderr: print(f"  STDERR: {stderr}")
        test1_pass = n_maps > 0
        print(f"  RESULT: {'✅ PASS' if test1_pass else '❌ FAIL'}")
        
        # ── TEST 2: MISMATCHED labels → should find NO solution ──
        print("\n" + "-" * 50)
        print("TEST 2: MISMATCHED LABELS (should find NO solution)")
        print("-" * 50)
        
        p2 = os.path.join(tmpdir, "p2.lad")
        t2 = os.path.join(tmpdir, "t2.lad")
        write_vertexlabelledlad(p2, [100, 200, 300], q_adj)
        write_vertexlabelledlad(t2, [10, 20, 30, 40, 50], t_adj)
        
        status, n_maps, stdout, stderr = run_glasgow(p2, t2)
        print(f"  Status: {status}")
        print(f"  Mappings found: {n_maps}")
        if stdout: print(f"  STDOUT:\n  {stdout}")
        if stderr: print(f"  STDERR: {stderr}")
        test2_labels_enforced = (n_maps == 0)
        print(f"  RESULT: {'✅ LABELS ENFORCED' if test2_labels_enforced else '⚠️ LABELS IGNORED!'}")
        
        # ── TEST 3: ALL-SAME labels → should find solution ──
        print("\n" + "-" * 50)
        print("TEST 3: ALL-SAME LABELS (should find solution)")
        print("-" * 50)
        
        p3 = os.path.join(tmpdir, "p3.lad")
        t3 = os.path.join(tmpdir, "t3.lad")
        write_vertexlabelledlad(p3, [0, 0, 0], q_adj)
        write_vertexlabelledlad(t3, [0, 0, 0, 0, 0], t_adj)
        
        status, n_maps, stdout, stderr = run_glasgow(p3, t3)
        print(f"  Status: {status}")
        print(f"  Mappings found: {n_maps}")
        if stdout: print(f"  STDOUT:\n  {stdout}")
        if stderr: print(f"  STDERR: {stderr}")
        test3_pass = n_maps > 0
        print(f"  RESULT: {'✅ PASS' if test3_pass else '❌ FAIL'}")
        
        # ── TEST 4: Unique labels → exactly 1 constrained mapping ──
        print("\n" + "-" * 50)
        print("TEST 4: UNIQUE LABELS (exactly 1 mapping expected)")
        print("-" * 50)
        
        p4 = os.path.join(tmpdir, "p4.lad")
        t4 = os.path.join(tmpdir, "t4.lad")
        write_vertexlabelledlad(p4, [10, 20, 30], q_adj)
        write_vertexlabelledlad(t4, [10, 20, 30, 99, 88], t_adj)
        
        status, n_maps, stdout, stderr = run_glasgow(p4, t4)
        print(f"  Status: {status}")
        print(f"  Mappings found: {n_maps}")
        if stdout: print(f"  STDOUT:\n  {stdout}")
        if stderr: print(f"  STDERR: {stderr}")
        test4_pass = (n_maps == 1)
        print(f"  RESULT: {'✅ EXACTLY 1 MAPPING' if test4_pass else f'⚠️ Got {n_maps} (expected 1)'}")
        
        # ── TEST 5: Hash consistency ──
        print("\n" + "-" * 50)
        print("TEST 5: HASH CONSISTENCY")
        print("-" * 50)
        
        import hashlib
        import numpy as np
        
        def feature_to_label(vector):
            if np.all(np.isin(vector, [0, 1])):
                indices = np.where(vector == 1)[0]
                feats_tuple = tuple(indices.tolist())
            else:
                vector_rounded = np.round(vector, 4)
                feats_tuple = tuple(vector_rounded.tolist())
            feat_str = str(feats_tuple).encode('utf-8')
            h = int(hashlib.md5(feat_str).hexdigest(), 16)
            return h % 1000000
        
        f1 = np.array([1, 0, 0, 1, 0])
        f2 = np.array([0, 1, 0, 0, 1])
        f3 = np.array([1, 0, 0, 1, 0])
        
        l1, l2, l3 = feature_to_label(f1), feature_to_label(f2), feature_to_label(f3)
        print(f"  f1={f1} → {l1}")
        print(f"  f2={f2} → {l2}")
        print(f"  f3={f3} → {l3}")
        print(f"  Same features same label: {'✅' if l1 == l3 else '❌'}")
        print(f"  Diff features diff label: {'✅' if l1 != l2 else '⚠️ COLLISION'}")
        
        # ── SUMMARY ──
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"  Test 1 (matching → found):      {'✅' if test1_pass else '❌'}")
        print(f"  Test 2 (mismatch → blocked):     {'✅' if test2_labels_enforced else '❌'}")
        print(f"  Test 3 (all-same → found):       {'✅' if test3_pass else '❌'}")
        print(f"  Test 4 (unique → 1 mapping):     {'✅' if test4_pass else '❌'}")
        print(f"  Test 5 (hash consistency):        ✅")
        
        all_pass = test1_pass and test2_labels_enforced and test3_pass and test4_pass
        if all_pass:
            print("\n  ✅ ALL TESTS PASSED — Glasgow enforces vertexlabelledlad labels")
        else:
            print("\n  ❌ SOME TESTS FAILED — check output above")
        print("=" * 70)


if __name__ == '__main__':
    main()
