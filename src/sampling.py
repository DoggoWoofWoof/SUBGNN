"""
Hierarchical Sampling Module for Training

This module provides SparseTensor-optimized query generation for fast GPU training.
Uses torch_sparse operations for O(1) neighbor lookups.

NOTE: For evaluation, use query_generator.py instead (NetworkX-based, more accurate).
The two modules serve different purposes:
- sampling.py: Fast training with SparseTensor (GPU-optimized)
- query_generator.py: Accurate evaluation with NetworkX (flexible, interpretable)
"""

import torch
import random
import sys
import time
from collections import deque, defaultdict, Counter
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

def _extract_fragment_fast_rw(source_graph, target_size):
    """Extract a connected fragment via random walk. Returns node indices or None."""
    # Defensive checks
    if source_graph is None:
        return None
    if not hasattr(source_graph, 'num_nodes') or source_graph.num_nodes == 0:
        return None
    if not hasattr(source_graph, 'edge_index') or source_graph.edge_index is None:
        return None
    if source_graph.edge_index.numel() == 0 or source_graph.num_edges == 0:
        return None
    
    device = source_graph.edge_index.device
    
    # Use cached adj_t if available
    if hasattr(source_graph, 'adj_t') and source_graph.adj_t is not None:
        adj = source_graph.adj_t
    else:
        adj = SparseTensor(row=source_graph.edge_index[0], col=source_graph.edge_index[1], 
                          sparse_sizes=(source_graph.num_nodes, source_graph.num_nodes))
    
    # CRITICAL: Get BOTH row_ptr and col from the SAME CSR representation
    # Mixing row_ptr from CSR with edge_index[1] (COO) causes segfaults!
    row_ptr, col_csr, _ = adj.csr()
    
    if row_ptr.numel() < 2 or col_csr.numel() == 0:
        return None
    
    max_node_idx = row_ptr.size(0) - 2
    if max_node_idx < 0:
        return None
    
    start_node = torch.randint(0, source_graph.num_nodes, (1,), device=device).item()
    if start_node > max_node_idx:
        return None
    
    try:
        # Use col_csr - NOT edge_index[1]! This is the critical fix.
        walk = torch.ops.torch_sparse.random_walk(
            row_ptr, col_csr,
            torch.tensor([start_node], device=device), target_size
        )[0]
        q_nodes = torch.unique(walk)
        return q_nodes if len(q_nodes) >= target_size / 2 else None
    except Exception as e:
        return None

def _extract_subgraph_from_adj(adj_t, node_indices, original_data):
    # Fast subgraph extraction using CSR slicing: O(num_subset_nodes * avg_degree)
    
    # 1. Slicing the SparseTensor to get the sub-adjacency
    # adj_t[rows, cols]
    
    # CRITICAL FIX: Ensure indices are on the correct devices for mixed CPU/GPU slicing
    
    # 1a. Slice STRUCTURE (GPU or CPU, depending on adj_t)
    adj_device = adj_t.device()
    if node_indices.device != adj_device:
        node_indices_struct = node_indices.to(adj_device)
    else:
        node_indices_struct = node_indices
        
    sub_adj = adj_t[node_indices_struct, node_indices_struct]
    row, col, _ = sub_adj.coo()
    edge_index = torch.stack([row, col], dim=0) # on adj_device
    
    # 1b. Slice FEATURES (CPU, as we moved data.x to CPU to save VRAM)
    feat_device = original_data.x.device
    if node_indices.device != feat_device:
        node_indices_feat = node_indices.to(feat_device)
    else:
        node_indices_feat = node_indices
        
    x = original_data.x[node_indices_feat]
    
    # 2. Return Data object on CPU (to avoid clogging GPU with batch queue)
    # DataLoader handles moving final batch to GPU
    data = Data(x=x.cpu(), edge_index=edge_index.cpu(), num_nodes=len(node_indices))
    
    # Note: node_indices are global IDs. The new edge_index is already re-indexed to 0..len(subset)-1 by the slicing!
    # data constructed above
    
    # Preserve necessary attributes if they exist
    if hasattr(original_data, 'node_type'):
        data.node_type = original_data.node_type[node_indices]
    if hasattr(original_data, 'global_id'):
        data.global_id = original_data.global_id[node_indices]
    else:
        # If original data doesn't have global_id, assign the absolute indices
        data.global_id = node_indices
        
    # CRITICAL: Preserve global metadata like node_types list to avoid KeyErrors during collation
    if hasattr(original_data, 'node_types'):
        data.node_types = original_data.node_types
    if hasattr(original_data, 'node_offset'):
        data.node_offset = original_data.node_offset
        
    return data

def _finalize_query_from_nodes(original_data, adj_t, global_node_indices, min_nodes):
    if not global_node_indices: return None, None
    
    # Ensure indices are a unique tensor on the correct device
    if isinstance(global_node_indices, list):
        q_global_nodes = torch.tensor(list(set(global_node_indices)), dtype=torch.long, device=original_data.x.device)
    else:
        q_global_nodes = torch.unique(global_node_indices)
        
    if len(q_global_nodes) < min_nodes: return None, None
    
    Gq = _extract_subgraph_from_adj(adj_t, q_global_nodes, original_data)
    return Gq, q_global_nodes

def are_partitions_neighbors_sparse(adj_t, nodes1, nodes2):
    """
    Checks neighbor connectivity using SparseTensor (CSR) slicing.
    adj_t: The global SparseTensor of the graph.
    nodes1, nodes2: Tensors of global node indices.
    """
    # Optimized: Use SparseTensor slicing which is implemented in C++ / CUDA
    sub_adj = adj_t[nodes1, nodes2]
    return sub_adj.nnz() > 0

def generate_multi_coarse_partition_query(original_data, adj_t, coarse_part_graph, fine_graphs, fine_part_nodes_map, fine_to_coarse_map, coarse_to_fine_map, possible_start_edges, coarse_edge_to_fine_bridges=None, min_nodes=80, max_nodes=100):
    if coarse_part_graph.number_of_edges() == 0: raise RuntimeError("Coarse graph has no edges.")
    configurations = [(2, 2),(3, 2),(4, 2),(3, 3),(4, 3),(5, 2),(5, 3),(5, 4),(6, 3),(6, 4),(6, 5),(8, 4),(8, 5),(8, 6),(10, 5),(10, 6),(10, 7),(12, 6),(12, 7),(12, 8),(15, 7),(15, 8),(15, 9)]; random.shuffle(configurations)
    
    # Pre-computed maps passed as arguments
    
    t_start_search = time.time()
    
    for num_frags, min_coarse_parts in configurations:
        random.shuffle(possible_start_edges) # Shuffle to randomize search start
        
        # Optimization: Use pre-computed valid bridges if available
        if coarse_edge_to_fine_bridges is not None:
            # Iterate through shuffled coarse edges (c1, c2)
            # We assume possible_start_edges are already edges in the coarse graph
            # We can also just iterate keys of coarse_edge_to_fine_bridges if we want, but possible_start_edges is fine
            
            for c_idx1, c_idx2 in possible_start_edges:
                # Get valid bridges for this coarse edge
                # The map might store (c1, c2) or (c2, c1), or both if we made it symmetric.
                # Assuming we made it symmetric or cover both directions.
                bridges = coarse_edge_to_fine_bridges.get((c_idx1, c_idx2))
                if not bridges:
                     bridges = coarse_edge_to_fine_bridges.get((c_idx2, c_idx1))
                
                if not bridges: continue
                
                # Pick a random valid bridge (f1, f2)
                f1, f2 = random.choice(bridges)
                
                # Guaranteed connected!
                # checks += 1 # Technically 1 check
                
                q_fine_indices, queue, visited = [f1, f2], [f1, f2], {f1, f2}
                # Standard BFS expansion to find connected fine partitions
                while queue and len(q_fine_indices) < num_frags:
                    current_fine_idx = queue.pop(0); current_c_idx = fine_to_coarse_map[current_fine_idx]
                    coarse_neighbors_and_self = list(coarse_part_graph.neighbors(current_c_idx)) + [current_c_idx]
                    potential_fine_neighbors = [fn for c_idx in coarse_neighbors_and_self for fn in coarse_to_fine_map.get(c_idx, [])]
                    random.shuffle(potential_fine_neighbors)
                    
                    for neighbor_idx in potential_fine_neighbors:
                        if neighbor_idx not in visited and are_partitions_neighbors_sparse(adj_t, fine_part_nodes_map[current_fine_idx], fine_part_nodes_map[neighbor_idx]):
                            visited.add(neighbor_idx); queue.append(neighbor_idx); q_fine_indices.append(neighbor_idx)
                            if len(q_fine_indices) >= num_frags: break
                            
                if len(q_fine_indices) < num_frags: continue
                true_coarse_indices = {fine_to_coarse_map[f_idx] for f_idx in q_fine_indices}
                if len(true_coarse_indices) < min_coarse_parts: continue
                
                nodes_per_frag = max_nodes // num_frags; all_query_nodes = []
                for fine_idx in q_fine_indices:
                    local_nodes = _extract_fragment_fast_rw(fine_graphs[fine_idx], nodes_per_frag)
                    if local_nodes is not None: 
                        all_query_nodes.extend(fine_part_nodes_map[fine_idx][local_nodes].tolist()) 
                
                Gq, _ = _finalize_query_from_nodes(original_data, adj_t, all_query_nodes, min_nodes)
                
                # Construct Gpos (stitched fine partitions) for consistency
                stitched_nodes = torch.cat([fine_part_nodes_map[idx] for idx in q_fine_indices])
                G_stitched = _extract_subgraph_from_adj(adj_t, stitched_nodes, original_data)
                
                duration = time.time()-t_start_search
                if duration > 5.0 and random.random() < 0.001:
                    print(f"[PROFILE] multi-coarse (optimized) match found after {duration:.4f}s.", file=sys.stderr)
                metadata = {'type': 'multi-coarse-opt', 'time': duration}
                return Gq, G_stitched, list(true_coarse_indices), metadata
        
        else:
             # Should not be reached if bridges are computed
             continue
                    
    raise RuntimeError("Failed to generate multi-coarse-partition query.")

def generate_hierarchical_sample(original_data, adj_t, coarse_graphs, fine_graphs, node_to_coarse_tensor, fine_to_coarse_map, coarse_to_fine_map, coarse_edges_list, fine_part_nodes_map, coarse_part_nodes_map, coarse_part_graph, k=3, q_size_min=20, q_size_max=120, prob_k_hop=0.2, prob_single_part=0.2, prob_multi_coarse=0.4, max_gpos_nodes=4000, coarse_edge_to_fine_bridges=None):
    
    t0 = time.time()
    rand_choice = random.random(); device = original_data.x.device; Gq, Gpos, G_coarse_pos = None, None, None
    sample_type = "unknown"

    if rand_choice < prob_k_hop:
        sample_type = "k-hop"
        t_0_khop = time.time()
        # 1. Anchor
        anchor = torch.randint(0, original_data.num_nodes, (1,), device=device).item()
        
        # 2. Positive Context Pool (k=6)
        subset_k_hop, _, _, _ = k_hop_subgraph(anchor, k, original_data.edge_index, relabel_nodes=False)
        
        if len(subset_k_hop) < q_size_min:
             # print(f"[DEBUG] k-hop failed: k-hop size {len(subset_k_hop)} < min {q_size_min}", file=sys.stderr)
             return None

        # 3. Query Sampling: Connected BFS Blob
        current_q_size = random.randint(q_size_min, q_size_max)
        
        if len(subset_k_hop) > current_q_size:
            query_nodes_list = [anchor]
            visited = {anchor}
            queue = deque([anchor])
            
            while len(query_nodes_list) < current_q_size and queue:
                u = queue.popleft()
                row, col, _ = adj_t[u].coo() 
                neighbors = col 
                
                for v_tn in neighbors:
                    v = v_tn.item()
                    if v not in visited:
                        visited.add(v)
                        query_nodes_list.append(v)
                        queue.append(v)
                        if len(query_nodes_list) >= current_q_size:
                            break
            
            query_nodes = torch.tensor(query_nodes_list, device=device)
        else:
            query_nodes = subset_k_hop

        # 4. Identify Coarse Partitions (Positive Context)
        subset_coarse_ids = node_to_coarse_tensor[query_nodes]
        mask = subset_coarse_ids >= 0
        if mask.sum() == 0: 
            # print("[DEBUG] k-hop failed: no valid coarse IDs found in query", file=sys.stderr)
            return None
            
        unique_coarse_ids, counts = torch.unique(subset_coarse_ids[mask], return_counts=True)
        
        # Optimization: Limit to top-10 interacting partitions to prevent OOM
        k_partitions = 10
        if len(unique_coarse_ids) > k_partitions:
            _, top_indices = torch.topk(counts, k_partitions)
            unique_coarse_ids = unique_coarse_ids[top_indices]

        # 5. Construct Gpos from these partitions
        target_device = original_data.x.device 
        pos_nodes_list = [coarse_part_nodes_map[cid.item()].to(target_device) for cid in unique_coarse_ids]
        if not pos_nodes_list: return None
        all_pos_nodes = torch.cat(pos_nodes_list).unique() 
        
        t_extract = time.time()
        try:
            # Manageable size now
            Gpos = _extract_subgraph_from_adj(adj_t, all_pos_nodes, original_data)
        except RuntimeError as e:
            if random.random() < 0.01:
                 print(f"[ERROR] k-hop Error during Gpos extraction (size {len(all_pos_nodes)}): {e}", file=sys.stderr)
            return None
            
        dur_extract = time.time() - t_extract
        
        # 6. Extract Gq 
        Gq = _extract_subgraph_from_adj(adj_t, query_nodes, original_data)
        
        # Representative coarse graph (mode)
        mode_id = torch.mode(subset_coarse_ids[mask]).values.item()
        
        # reconstruct coarse graph from original data to get features (as coarse_graphs[mode_id] now lacks them)
        G_coarse_pos = _extract_subgraph_from_adj(adj_t, coarse_part_nodes_map[mode_id], original_data)
        
        dur_khop = time.time() - t_0_khop
        # Updated profile log
        if dur_khop > 5.0 and random.random() < 0.001:
            print(f"[PROFILE] k-hop({k}) total:{dur_khop:.4f}s (k_hop_pool:{len(subset_k_hop)}, query_size:{len(query_nodes)}, pos_size:{len(all_pos_nodes)}, parts:{len(unique_coarse_ids)})", file=sys.stderr)
        metadata = {'type': 'k-hop', 'time': dur_khop}

    elif rand_choice < prob_k_hop + prob_single_part:
        sample_type = "single-part"
        t_single = time.time()
        
        if not fine_graphs: return None
        fine_idx = random.choice(list(fine_to_coarse_map.keys())); Gpos = fine_graphs[fine_idx]
        
        # Relaxed check or kept? Keeping check for single-part as it should be small.
        if Gpos.num_nodes > max_gpos_nodes: 
             # print(f"[DEBUG] single-part failed: size {Gpos.num_nodes} > max", file=sys.stderr)
             return None
        
        q_nodes_local = _extract_fragment_fast_rw(Gpos, random.randint(q_size_min, q_size_max))
        if q_nodes_local is None: return None
        
        q_mask = torch.zeros(Gpos.num_nodes, dtype=torch.bool, device=device); q_mask[q_nodes_local] = True
        Gq = Gpos.subgraph(q_mask)
        
        # If Gpos (fine graph) lacks features, Gq will too.
        # We need to reconstruct Gq and Gpos with features.
        pos_nodes_global = fine_part_nodes_map[fine_idx]
        Gpos = _extract_subgraph_from_adj(adj_t, pos_nodes_global, original_data)
        
        q_nodes_global = pos_nodes_global[q_nodes_local]
        Gq = _extract_subgraph_from_adj(adj_t, q_nodes_global, original_data)
        
        coarse_parent_idx = fine_to_coarse_map.get(fine_idx)
        if coarse_parent_idx is None: return None
        # reconstruct coarse graph with features
        G_coarse_pos = _extract_subgraph_from_adj(adj_t, coarse_part_nodes_map[coarse_parent_idx], original_data)
        
        duration = time.time() - t_single
        if duration > 5.0 and random.random() < 0.001:
            print(f"[PROFILE] single-part took {duration:.4f}s", file=sys.stderr)
        metadata = {'type': 'single-part', 'time': duration}
        
    elif rand_choice < prob_k_hop + prob_single_part + prob_multi_coarse:
        sample_type = "multi-coarse"
        t_multi = time.time()
        try:
            res = generate_multi_coarse_partition_query(original_data, adj_t, coarse_part_graph, fine_graphs, fine_part_nodes_map, fine_to_coarse_map, coarse_to_fine_map, coarse_edges_list, coarse_edge_to_fine_bridges=coarse_edge_to_fine_bridges, min_nodes=q_size_min, max_nodes=q_size_max)
            
            if res is None: return None
            Gq, Gpos, coarse_indices, meta_mc = res
            metadata = meta_mc # Already computed
                 
            all_coarse_pos_nodes = torch.cat([coarse_part_nodes_map[c_idx] for c_idx in coarse_indices])
            G_coarse_pos = _extract_subgraph_from_adj(adj_t, all_coarse_pos_nodes, original_data)
            
            duration = time.time() - t_multi
            if duration > 5.0 and random.random() < 0.001:
                print(f"[PROFILE] multi-coarse took {duration:.4f}s", file=sys.stderr)
        except RuntimeError: return None
        
    else:
        sample_type = "sibling-walk"
        t_walk = time.time()
        if not fine_part_nodes_map or len(fine_part_nodes_map) < 2: return None
        
        Gpos = None
        source_part_indices = None
        
        for attempt in range(10):
            num_frags = random.randint(2, 3); start_fine_idx = random.choice(list(fine_part_nodes_map.keys())); coarse_parent_idx = fine_to_coarse_map.get(start_fine_idx)
            if coarse_parent_idx is None: continue
            
            siblings = [idx for idx, c_idx in fine_to_coarse_map.items() if c_idx == coarse_parent_idx]
            if len(siblings) < num_frags: continue

            source_part_indices = {start_fine_idx}; queue = [start_fine_idx]
            random.shuffle(siblings) 
            
            # Optimization: Try to find neighbors among siblings actively
            potential_neighbors = [s for s in siblings if s != start_fine_idx]
            random.shuffle(potential_neighbors)
            
            current_cluster = [start_fine_idx]
            
            for candidate in potential_neighbors:
                is_connected = False
                for node in current_cluster:
                     if are_partitions_neighbors_sparse(adj_t, fine_part_nodes_map[node], fine_part_nodes_map[candidate]):
                         is_connected = True
                         break
                
                if is_connected:
                    current_cluster.append(candidate)
                    if len(current_cluster) >= num_frags: break
            
            if len(current_cluster) >= num_frags:
                source_part_indices = set(current_cluster)
                pos_nodes = torch.cat([fine_part_nodes_map[i] for i in source_part_indices])
                if len(pos_nodes) <= max_gpos_nodes:
                     Gpos = _extract_subgraph_from_adj(adj_t, pos_nodes, original_data)
                     break

        if Gpos is None:
            return None
        
        nodes_per_frag = (q_size_min + q_size_max) // (2 * num_frags); all_query_global_nodes = []
        for fine_idx in source_part_indices:
            local_indices = _extract_fragment_fast_rw(fine_graphs[fine_idx], nodes_per_frag)
            if local_indices is not None:
                all_query_global_nodes.extend(fine_part_nodes_map[fine_idx][local_indices].tolist())
            
        Gq, _ = _finalize_query_from_nodes(original_data, adj_t, all_query_global_nodes, min_nodes=q_size_min)
        if Gq is None: return None
        G_coarse_pos = _extract_subgraph_from_adj(adj_t, coarse_part_nodes_map[coarse_parent_idx], original_data)
        
        duration = time.time() - t_walk
        if duration > 5.0 and random.random() < 0.001:
            print(f"[PROFILE] sibling-walk took {duration:.4f}s", file=sys.stderr)
        metadata = {'type': 'sibling-walk', 'time': duration}
        
    if Gq is None or Gpos is None or G_coarse_pos is None: return None
    if 'metadata' not in locals(): metadata = {'type': sample_type, 'time': time.time()-t0}
    return Gq, Gpos, G_coarse_pos, metadata
