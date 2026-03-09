#!/bin/bash
# Patch EvaluateQuery.cpp to print each embedding found.
# The enumeration functions all use: embedding[query_vertex] = target_vertex
# and increment embedding_count when a match is found.
# We add a print statement right after each embedding_count++ to output the mapping.

FILE="/app/SubgraphMatching/matching/EvaluateQuery.cpp"

# Create a helper C++ function that prints an embedding
# Insert it at the top of the file after the includes
sed -i '/^#include "EvaluateQuery.h"/a \
\n// Print embedding mapping: query_node -> target_node\nstatic void print_embedding(const ui* embedding, ui query_vertices_num, size_t embedding_count) {\n    printf("Embedding %zu:", embedding_count);\n    for (ui i = 0; i < query_vertices_num; ++i) {\n        printf(" %u", embedding[i]);\n    }\n    printf("\\n");\n    fflush(stdout);\n}' "$FILE"

# Now find every "embedding_count += 1;" and add a print call after it.
# The enumeration functions use either "embedding_count += 1;" or "embedding_count++;"
# and the embedding array + query vertex count are always in scope.
# 
# For the functions that use query_vertices_num:
#   exploreGraph, LFTJ, exploreGraphQLStyle, exploreQuickSIStyle
# For the DPiso functions:
#   exploreDPisoStyle, exploreDPisoRecursiveStyle
# 
# We use a sed approach: after each "embedding_count += 1;" line, insert the print.

sed -i '/embedding_count += 1;/a \
                print_embedding(embedding, query_vertices_num, embedding_count);' "$FILE"

echo "Patch applied. Embedding printing enabled."
cat "$FILE" | grep -n "print_embedding" | head -20
