"""
Patch EvaluateQuery.cpp to print actual embeddings.

Usage: python3 patch_embeddings.py /path/to/EvaluateQuery.cpp

After patching, each embedding will be printed as:
    Embedding N: v0 v1 v2 ...
where position i = query vertex i, value = target vertex.

Variable names used in the C++ code:
    - embedding[] array: embedding[query_vertex] = target_vertex
    - embedding_cnt: counter
    - max_depth: number of query vertices (= query_vertices_num)
    - order[]: mapping from depth to query vertex
"""
import sys

def patch(filepath):
    code = open(filepath).read()

    # 1. Add print_embedding function after includes
    print_fn = """
// --- PATCH: Print each found embedding ---
static void print_embedding(const ui* embedding, int max_depth, const ui* order, size_t cnt) {
    // Print in query-vertex order: embedding[order[0]], embedding[order[1]], ...
    // But we want position i = query vertex i, so use embedding directly
    printf("Embedding %zu:", cnt);
    for (int i = 0; i < max_depth; ++i) {
        printf(" %u", embedding[i]);
    }
    printf("\\n");
    fflush(stdout);
}
// --- END PATCH ---
"""
    anchor = '#include "EvaluateQuery.h"'
    if anchor in code:
        code = code.replace(anchor, anchor + print_fn, 1)
    else:
        print(f"WARNING: Could not find '{anchor}' in {filepath}")
        return False

    # 2. After each 'embedding_cnt += 1;', add a print call
    # The embedding array is indexed by query vertex ID directly: embedding[u] = v
    # max_depth = query_graph->getVerticesCount()
    old = 'embedding_cnt += 1;'
    new = 'embedding_cnt += 1;\n                print_embedding(embedding, max_depth, order, embedding_cnt);'
    count = code.count(old)
    if count == 0:
        print(f"WARNING: No occurrences of '{old}' found")
        return False

    code = code.replace(old, new)
    open(filepath, 'w').write(code)
    print(f"Patched {count} embedding print points in {filepath}")
    return True


if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else '/app/SubgraphMatching/matching/EvaluateQuery.cpp'
    success = patch(filepath)
    sys.exit(0 if success else 1)
