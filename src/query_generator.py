import itertools
import random
from collections import defaultdict
from typing import List, Optional, Set, Tuple

import networkx as nx
import torch
import sys
from collections import defaultdict, deque
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph

from src.config import DEVICE
from src.utils import are_partitions_neighbors

# Configs for multi-coarse queries: (num_fragments, min_coarse_parts)
MULTI_COARSE_CONFIGS = [
    (2, 2), (3, 2), (4, 2), (3, 3), (4, 3), (5, 2), (5, 3), (5, 4),
    (6, 3), (6, 4), (6, 5), (8, 4), (8, 5), (8, 6), (10, 5), (10, 6)
]

def _extract_fragment(graph: Data, target_nodes: int) -> Optional[List[int]]:
    """
    Extracts a connected component fragment of a target size from a graph
    using a Breadth-First Search (BFS) starting from a random node.
    """
    if graph.num_nodes < 5 or target_nodes < 5:
        return None

    # Conversion to NX can be slow for huge graphs, but partitions are "fine" (small) so this is okay.
    graph_nx = to_networkx(graph, to_undirected=True)
    
    if not nx.is_connected(graph_nx):
        if not list(graph_nx.nodes()):
            return None
        largest_cc = max(nx.connected_components(graph_nx), key=len)
        graph_nx = graph_nx.subgraph(largest_cc)

    if graph_nx.number_of_nodes() == 0:
        return None

    nodes_list = list(graph_nx.nodes())
    if not nodes_list: return None
    
    start_node = random.choice(nodes_list)
    queue, visited, fragment = [start_node], {start_node}, [start_node]

    while queue and len(fragment) < target_nodes:
        current = queue.pop(0)
        neighbors = list(graph_nx.neighbors(current))
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                fragment.append(neighbor)
                if len(fragment) >= target_nodes:
                    break
    return fragment


def _finalize_query_from_nodes(
    original_data: Data, global_nodes: List[int], min_nodes: int, device: torch.device
) -> Tuple[Optional[Data], Optional[List[int]]]:
    """
    Takes a list of global node IDs, creates a subgraph, finds the largest
    connected component, and returns it if it's large enough.
    """
    if not global_nodes:
        return None, None

    unique_global_nodes = sorted(list(set(global_nodes)))
    # Subgraph extraction
    temp_graph = original_data.subgraph(
        torch.tensor(unique_global_nodes, device=device)
    )
    temp_nx = to_networkx(temp_graph, to_undirected=True)

    if temp_nx.number_of_nodes() == 0:
        return None, None

    # Ensure the query is a single connected component
    largest_cc_nodes = (
        max(nx.connected_components(temp_nx), key=len)
        if not nx.is_connected(temp_nx)
        else list(temp_nx.nodes())
    )

    if len(largest_cc_nodes) < min_nodes:
        return None, None

    # Map back to index in unique_global_nodes
    final_global_nodes = [unique_global_nodes[i] for i in largest_cc_nodes]
    
    # Create final PyG object
    Gq = original_data.subgraph(torch.tensor(final_global_nodes, device=device))
    return Gq, final_global_nodes


def generate_single_partition_query(
    **kwargs,
) -> Tuple[Data, Data, List[int], List[int], Set[int]]:
    """
    Generates a query that is entirely contained within a single fine-level partition.
    """
    fine_graphs, fine_to_coarse_map, device = (
        kwargs["fine_graphs"],
        kwargs["fine_to_coarse_map"],
        kwargs["device"],
    )
    min_nodes, max_nodes = kwargs["min_nodes"], kwargs["max_nodes"]
    anchor_coarse_idx = kwargs.get("anchor_coarse_idx", None)

    Gq = None
    valid_indices = list(range(len(fine_graphs)))
    if anchor_coarse_idx is not None:
        valid_indices = [
            idx for idx in valid_indices if fine_to_coarse_map[idx] == anchor_coarse_idx
        ]

    attempts = 0
    while Gq is None and attempts < 20:
        attempts += 1
        if not valid_indices:
            # raise RuntimeError("No fine partitions in anchor coarse partition.")
            return None # Graceful fail
            
        true_fine_idx = random.choice(valid_indices)
        anchor_partition = fine_graphs[true_fine_idx]
        
        # Determine target size
        target_size = random.randint(min_nodes, max_nodes)
        if target_size > anchor_partition.num_nodes: target_size = anchor_partition.num_nodes

        local_nodes = _extract_fragment(anchor_partition, target_size)

        if local_nodes and len(local_nodes) >= min_nodes:
            # We need global IDs
            # fine_part_nodes_map maps PartID -> Tensor of Global IDs
            global_ids_map = kwargs["fine_part_nodes_map"][true_fine_idx]
            q_global_nodes = [global_ids_map[i].item() for i in local_nodes]
            
            # Re-extract from original to get features/edges correct (especially if partition graph was simplified)
            Gq, q_global_nodes = _finalize_query_from_nodes(kwargs["original_data"], q_global_nodes, min_nodes, device)
            
            if Gq is not None:
                true_coarse_idx = fine_to_coarse_map[true_fine_idx]
                return Gq, anchor_partition, q_global_nodes, [true_fine_idx], {true_coarse_idx}

    return None # Failed


def generate_multi_fine_partition_query(
    **kwargs,
) -> Tuple[Data, Data, List[int], List[int], Set[int]]:
    """
    Generates a query by stitching together fragments from multiple neighboring
    fine-partitions that all belong to the *same* coarse partition.
    """
    G_nx, fine_graphs, fine_part_nodes_map, fine_to_coarse_map, device = (
        kwargs["G_nx"],
        kwargs["fine_graphs"],
        kwargs["fine_part_nodes_map"],
        kwargs["fine_to_coarse_map"],
        kwargs["device"],
    )
    num_frags, min_nodes, max_nodes = (
        kwargs["num_frags"],
        kwargs["min_nodes"],
        kwargs["max_nodes"],
    )
    anchor_coarse_idx = kwargs.get("anchor_coarse_idx", None)

    for _ in range(50):  # Try 50 times
        candidate_starts = (
            [idx for idx, c in fine_to_coarse_map.items() if c == anchor_coarse_idx]
            if anchor_coarse_idx is not None
            else list(fine_part_nodes_map.keys())
        )
        if not candidate_starts: return None

        start_fine_idx = random.choice(candidate_starts)
        true_coarse_idx = fine_to_coarse_map[start_fine_idx]

        siblings = [
            idx for idx, c_idx in fine_to_coarse_map.items() if c_idx == true_coarse_idx
        ]

        # BFS on Partition Graph (implied by G_nx connectivity)
        q_fine_indices, queue, visited = (
            [start_fine_idx],
            [start_fine_idx],
            {start_fine_idx},
        )
        
        while queue and len(q_fine_indices) < num_frags:
            current_idx = queue.pop(0)
            random.shuffle(siblings)
            for neighbor_idx in siblings:
                if neighbor_idx not in visited and are_partitions_neighbors(
                    G_nx,
                    fine_part_nodes_map[current_idx],
                    fine_part_nodes_map[neighbor_idx],
                ):
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)
                    q_fine_indices.append(neighbor_idx)
                    if len(q_fine_indices) >= num_frags:
                        break

        if len(q_fine_indices) < num_frags:
            continue

        nodes_per_frag = max_nodes // num_frags
        all_query_nodes = []
        for fine_idx in q_fine_indices:
            local_nodes = _extract_fragment(fine_graphs[fine_idx], nodes_per_frag)
            if local_nodes:
                all_query_nodes.extend(
                    [fine_part_nodes_map[fine_idx][i].item() for i in local_nodes]
                )

        Gq, q_global_nodes = _finalize_query_from_nodes(
            kwargs["original_data"], all_query_nodes, min_nodes, device
        )
        if Gq:
            # "G_stitched" is the union of the true partitions (Ground Truth Context)
            # Actually we can just return the list of nodes or subgraph
            stitched_nodes = [
                node.item() for idx in q_fine_indices for node in fine_part_nodes_map[idx]
            ]
            G_stitched = kwargs["original_data"].subgraph(
                torch.tensor(stitched_nodes, device=device)
            )
            return Gq, G_stitched, q_global_nodes, q_fine_indices, {true_coarse_idx}

    return None


def generate_multi_coarse_partition_query(
    **kwargs,
) -> Tuple[Data, Data, List[int], List[int], Set[int]]:
    """
    Generates the most complex query type, stitching fragments from fine partitions
    that span across multiple, neighboring coarse partitions.
    """
    (
        original_data,
        G_nx,
        coarse_part_graph,
        fine_graphs,
        fine_part_nodes_map,
        fine_to_coarse_map,
        device,
    ) = (
        kwargs["original_data"],
        kwargs["G_nx"],
        kwargs["coarse_part_graph"],
        kwargs["fine_graphs"],
        kwargs["fine_part_nodes_map"],
        kwargs["fine_to_coarse_map"],
        kwargs["device"],
    )
    min_nodes, max_nodes = kwargs["min_nodes"], kwargs["max_nodes"]
    anchor_coarse_idx = kwargs.get("anchor_coarse_idx", None)

    if coarse_part_graph.number_of_edges() == 0:
        return None

    coarse_to_fine_map = defaultdict(list)
    for f_idx, c_idx in fine_to_coarse_map.items():
        coarse_to_fine_map[c_idx].append(f_idx)

    # Get candidate edges from anchor
    possible_edges = list(coarse_part_graph.edges())
    if anchor_coarse_idx is not None:
        possible_edges = [
            (u, v) for (u, v) in possible_edges
            if u == anchor_coarse_idx or v == anchor_coarse_idx
        ]
    
    if not possible_edges:
        return None
    random.shuffle(possible_edges)

    for num_frags, min_coarse_parts in MULTI_COARSE_CONFIGS:
        for c_idx1, c_idx2 in possible_edges[:10]:
            
            fine_parts_in_c1 = coarse_to_fine_map.get(c_idx1, [])
            fine_parts_in_c2 = coarse_to_fine_map.get(c_idx2, [])
            if not fine_parts_in_c1 or not fine_parts_in_c2:
                continue
            
            # Find connected boundary fine partitions
            f1, f2 = None, None
            for f1_cand in fine_parts_in_c1:
                for f2_cand in fine_parts_in_c2:
                    if are_partitions_neighbors(G_nx, fine_part_nodes_map[f1_cand], fine_part_nodes_map[f2_cand]):
                        f1, f2 = f1_cand, f2_cand
                        break
                if f1 is not None:
                    break
            
            if f1 is None:
                continue
                 
            # BFS expansion to get more fine partitions
            q_fine_indices = [f1, f2]
            visited = {f1, f2}
            queue = deque([f1, f2])
            
            while queue and len(q_fine_indices) < num_frags:
                current = queue.popleft()
                current_coarse = fine_to_coarse_map[current]
                
                # Get neighboring coarse partitions
                neighbor_coarse = list(coarse_part_graph.neighbors(current_coarse)) + [current_coarse]
                candidate_fines = [f for c in neighbor_coarse for f in coarse_to_fine_map.get(c, [])]
                random.shuffle(candidate_fines)
                
                for neighbor_fine in candidate_fines:
                    if neighbor_fine not in visited:
                        if are_partitions_neighbors(G_nx, fine_part_nodes_map[current], fine_part_nodes_map[neighbor_fine]):
                            visited.add(neighbor_fine)
                            queue.append(neighbor_fine)
                            q_fine_indices.append(neighbor_fine)
                            if len(q_fine_indices) >= num_frags:
                                break
            
            if len(q_fine_indices) < 2:
                continue
            
            # Check coarse coverage
            true_coarse_indices = {fine_to_coarse_map[f_idx] for f_idx in q_fine_indices}
            if len(true_coarse_indices) < min_coarse_parts:
                continue
            
            # Extract query nodes
            nodes_per_frag = max_nodes // max(2, len(q_fine_indices))
            all_query_nodes = []
            for fine_idx in q_fine_indices:
                local_nodes = _extract_fragment(fine_graphs[fine_idx], nodes_per_frag)
                if local_nodes:
                    all_query_nodes.extend([fine_part_nodes_map[fine_idx][i].item() for i in local_nodes])
            
            Gq, q_global_nodes = _finalize_query_from_nodes(original_data, all_query_nodes, min_nodes, device)
            
            if Gq:
                stitched_nodes = [node.item() for idx in q_fine_indices for node in fine_part_nodes_map[idx]]
                G_stitched = original_data.subgraph(torch.tensor(stitched_nodes, device=device))
                return Gq, G_stitched, q_global_nodes, q_fine_indices, true_coarse_indices
                 
    return None

