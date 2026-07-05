import modal
import multiprocessing
import copy
import sys
import time
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import itertools
import gc
from collections import Counter, defaultdict, deque


# Optional torch import for local testing without Modal
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    # Force single-threaded C++ backend for sparse operations
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None

# =============================================================================
# GRAPH PARTITIONING HELPERS (Global scope for multiprocessing pickling)
# =============================================================================

def _partitioner_target(queue, n_parts, xadj, adjncy):
    import pymetis
    try:
        _, membership = pymetis.part_graph(n_parts, xadj=xadj, adjncy=adjncy)
        queue.put(membership)
    except Exception as e:
        print(f"      - [Subprocess] Error during partitioning: {e}", flush=True)
        queue.put(None)

def _extract_fragment_fast_rw(source_graph, target_size):
    """Extract a connected fragment via random walk. Returns node indices or None."""
    from torch_sparse import SparseTensor
    
    # Defensive checks for empty/invalid graphs
    if source_graph is None or source_graph.num_nodes == 0 or source_graph.num_edges == 0:
        return None
    if not hasattr(source_graph, 'edge_index') or source_graph.edge_index is None:
        return None
    if source_graph.edge_index.numel() == 0:
        return None
        
    device = source_graph.edge_index.device
    
    # Use cached adj_t if available, otherwise create on-the-fly
    if hasattr(source_graph, 'adj_t') and source_graph.adj_t is not None:
        adj = source_graph.adj_t
    else:
        adj = SparseTensor(row=source_graph.edge_index[0], col=source_graph.edge_index[1], 
                          sparse_sizes=(source_graph.num_nodes, source_graph.num_nodes))
    
    row_ptr, col_csr, _ = adj.csr()
    
    if row_ptr.numel() < 2 or col_csr.numel() == 0: 
        return None
        
    # ---------------------------------------------------------
    # THE SIGSEGV FIX: 
    # Prevent torch_sparse C++ from crashing due to zero-degree start nodes.
    # ---------------------------------------------------------
    # Calculate out-degree for every node efficiently
    degrees = row_ptr[1:] - row_ptr[:-1]
    
    # Get indices of nodes that actually have edges
    valid_start_nodes = torch.nonzero(degrees > 0).view(-1)
    
    if valid_start_nodes.numel() == 0:
        return None # Graph has no valid edges to walk
        
    # Safely pick a random starting node from the valid pool
    random_idx = torch.randint(0, valid_start_nodes.numel(), (1,), device=device)
    start_node = valid_start_nodes[random_idx].item()

    try:
        walk = torch.ops.torch_sparse.random_walk(
            row_ptr, col_csr,
            torch.tensor([start_node], device=device), target_size
        )[0]
        q_nodes = torch.unique(walk)
        return q_nodes if len(q_nodes) >= target_size / 2 else None
    except Exception as e:
        print(f"      - [WARN] Random walk failed: {e}", flush=True)
        return None

def _bounded_node_sample(node_indices, max_nodes, required_nodes=None):
    """Bound a node set while preserving required/query nodes whenever possible."""
    node_indices = torch.unique(node_indices)
    if required_nodes is not None:
        required_nodes = torch.unique(required_nodes.to(node_indices.device))
        node_indices = torch.unique(torch.cat([node_indices, required_nodes]))

    if node_indices.numel() <= max_nodes:
        return node_indices

    if required_nodes is None or required_nodes.numel() == 0:
        perm = torch.randperm(node_indices.numel(), device=node_indices.device)
        return node_indices[perm[:max_nodes]]

    required_nodes = torch.unique(required_nodes.to(node_indices.device))
    if required_nodes.numel() >= max_nodes:
        perm = torch.randperm(required_nodes.numel(), device=required_nodes.device)
        return required_nodes[perm[:max_nodes]]

    non_required = node_indices[~torch.isin(node_indices, required_nodes)]
    take = max_nodes - required_nodes.numel()
    if non_required.numel() > take:
        perm = torch.randperm(non_required.numel(), device=non_required.device)
        non_required = non_required[perm[:take]]
    return torch.unique(torch.cat([required_nodes, non_required]))

def _extract_subgraph_from_adj(adj_t, node_indices, original_data, max_nodes=2000, preserve_nodes=None):
    """Fast subgraph extraction using SparseTensor slicing. Returns Data or None."""
    from torch_geometric.data import Data
    
    # HARD GUARD: SparseTensor 2D indexing is O(N²) — never allow large inputs
    node_indices = _bounded_node_sample(node_indices, max_nodes, preserve_nodes)
    
    # Align indices to adj_t device
    adj_device = adj_t.device()
    node_indices_struct = node_indices.to(adj_device) if node_indices.device != adj_device else node_indices
    
    # Defensive checks to prevent C++ crashes
    if node_indices_struct.numel() == 0:
        return None
    if node_indices_struct.max().item() >= adj_t.sparse_sizes()[0]:
        return None
    if node_indices_struct.min().item() < 0:
        return None

    try:
        sub_adj = adj_t[node_indices_struct, node_indices_struct]
        row, col, _ = sub_adj.coo()
        edge_index = torch.stack([row, col], dim=0)
    except Exception as e:
        print(f"      - [WARN] Adj slicing failed: {e}", flush=True)
        return None
    
    # Slice features (may be on different device)
    feat_device = original_data.x.device
    node_indices_feat = node_indices.to(feat_device) if node_indices.device != feat_device else node_indices
    x = original_data.x[node_indices_feat]
    
    # Build Data object on CPU for DataLoader
    data = Data(x=x.cpu(), edge_index=edge_index.cpu(), num_nodes=len(node_indices))
    
    # Preserve required attributes
    if hasattr(original_data, 'node_type'):
        data.node_type = original_data.node_type[node_indices]
    data.global_id = original_data.global_id[node_indices] if hasattr(original_data, 'global_id') else node_indices
    if hasattr(original_data, 'node_types'):
        data.node_types = original_data.node_types
    if hasattr(original_data, 'node_offset'):
        data.node_offset = original_data.node_offset
        
    return data

def _finalize_query_from_nodes(original_data, adj_t, global_node_indices, min_nodes):
    """Convert node indices to a query subgraph. Returns (Gq, nodes) or (None, None)."""
    if not global_node_indices:
        return None, None
    
    if isinstance(global_node_indices, list):
        q_global_nodes = torch.tensor(list(set(global_node_indices)), dtype=torch.long, device=original_data.x.device)
    else:
        q_global_nodes = torch.unique(global_node_indices)
        
    if len(q_global_nodes) < min_nodes:
        return None, None
    
    return _extract_subgraph_from_adj(adj_t, q_global_nodes, original_data), q_global_nodes

def _sample_bounded_k_hop_nodes(adj_t, anchor, max_hops, target_size, max_neighbors_per_node=64, device=None):
    """Sample a connected query inside K hops without materializing the full K-hop ball."""
    row_ptr, col_indices, _ = adj_t.csr()
    num_nodes = row_ptr.size(0) - 1
    anchor = int(anchor)
    if anchor < 0 or anchor >= num_nodes:
        return None

    visited = {anchor}
    query_nodes = [anchor]
    queue = deque([(anchor, 0)])

    while queue and len(query_nodes) < target_size:
        u, depth = queue.popleft()
        if depth >= max_hops:
            continue

        start_ptr = int(row_ptr[u].item())
        end_ptr = int(row_ptr[u + 1].item())
        if end_ptr <= start_ptr:
            continue

        neighbors = col_indices[start_ptr:end_ptr]
        if neighbors.numel() > max_neighbors_per_node:
            perm = torch.randperm(neighbors.numel(), device=neighbors.device)[:max_neighbors_per_node]
            neighbors = neighbors[perm]
        elif neighbors.numel() > 1:
            perm = torch.randperm(neighbors.numel(), device=neighbors.device)
            neighbors = neighbors[perm]

        for v_tensor in neighbors:
            v = int(v_tensor.item())
            if v in visited:
                continue
            visited.add(v)
            query_nodes.append(v)
            queue.append((v, depth + 1))
            if len(query_nodes) >= target_size:
                break

    if len(query_nodes) < target_size:
        return None
    return torch.tensor(query_nodes, dtype=torch.long, device=device or col_indices.device)

def _sample_full_graph_random_walk_nodes(adj_t, target_size, device=None, attempts=20):
    """Sample a connected query by deduplicating a bounded random walk."""
    row_ptr, col_indices, _ = adj_t.csr()
    device = device or row_ptr.device
    degrees = row_ptr[1:] - row_ptr[:-1]
    valid_start_nodes = torch.nonzero(degrees > 0).view(-1)
    if valid_start_nodes.numel() == 0:
        return None

    min_nodes = max(5, target_size // 2)
    walk_len = max(target_size * 4, target_size + 16)
    for _ in range(attempts):
        random_idx = torch.randint(0, valid_start_nodes.numel(), (1,), device=device)
        start_node = valid_start_nodes[random_idx].view(1).to(device)
        try:
            walk = torch.ops.torch_sparse.random_walk(row_ptr, col_indices, start_node, walk_len)[0]
        except Exception:
            continue

        seen = set()
        ordered = []
        for node in walk.tolist():
            node = int(node)
            if node < 0 or node in seen:
                continue
            seen.add(node)
            ordered.append(node)
            if len(ordered) >= target_size:
                break
        if len(ordered) >= min_nodes:
            return torch.tensor(ordered, dtype=torch.long, device=device)
    return None

def _sample_degree_k_hop_nodes(adj_t, target_size, device=None, max_hops=3):
    """Bias k-hop sampling toward high-degree anchors for topology-heavy training examples."""
    row_ptr, _, _ = adj_t.csr()
    device = device or row_ptr.device
    degrees = row_ptr[1:] - row_ptr[:-1]
    nonzero = torch.nonzero(degrees > 0).view(-1)
    if nonzero.numel() == 0:
        return None

    pool_size = min(4096, nonzero.numel())
    if nonzero.numel() > pool_size:
        _, top_local = torch.topk(degrees[nonzero], pool_size)
        pool = nonzero[top_local]
    else:
        pool = nonzero

    for _ in range(12):
        anchor = int(pool[torch.randint(0, pool.numel(), (1,), device=device)].item())
        nodes = _sample_bounded_k_hop_nodes(adj_t, anchor, max_hops, target_size, device=device)
        if nodes is not None and nodes.numel() >= max(5, target_size // 2):
            return nodes
    return None

def are_fine_partitions_connected(f1, f2, coarse_edge_to_fine_bridges, fine_to_coarse_map):
    """Check if two fine partitions are connected using pre-computed bridges or coarse graph."""
    if f1 == f2: return True
    if coarse_edge_to_fine_bridges is None: return False
    c1 = fine_to_coarse_map.get(f1)
    c2 = fine_to_coarse_map.get(f2)
    if c1 is None or c2 is None: return False
    if c1 == c2: return True # Conservative: assume connectivity within same coarse partition
    bridges = coarse_edge_to_fine_bridges.get((c1, c2)) or coarse_edge_to_fine_bridges.get((c2, c1))
    if not bridges: return False
    return any((f1 == b[0] and f2 == b[1]) or (f1 == b[1] and f2 == b[0]) for b in bridges)

def are_partitions_neighbors_sparse(adj_t, nodes1, nodes2):
    """Check if two node sets share any edges via SparseTensor slicing."""
    adj_device = adj_t.device()
    if nodes1.device != adj_device:
        nodes1 = nodes1.to(adj_device)
    if nodes2.device != adj_device:
        nodes2 = nodes2.to(adj_device)
    
    # Bounds check
    if nodes1.numel() > 0 and nodes1.max() >= adj_t.sparse_sizes()[0]:
        return False
    if nodes2.numel() > 0 and nodes2.max() >= adj_t.sparse_sizes()[0]:
        return False

    try:
        return adj_t[nodes1, nodes2].nnz() > 0
    except Exception:
        return False

def generate_multi_coarse_partition_query(original_data, adj_t, coarse_part_graph, fine_graphs,
                                          fine_part_nodes_map, fine_to_coarse_map, coarse_to_fine_map, 
                                          possible_start_edges, coarse_edge_to_fine_bridges=None, 
                                          min_nodes=80, max_nodes=100):
    """Generate a query spanning multiple coarse partitions using pre-computed bridges."""
    if coarse_part_graph.number_of_edges() == 0:
        raise RuntimeError("Coarse graph has no edges.")
    
    # (num_fragments, min_coarse_partitions) configurations
    configurations = [
        (2, 2), (3, 2), (4, 2), (3, 3), (4, 3), (5, 2), (5, 3), (5, 4),
        (6, 3), (6, 4), (6, 5), (8, 4), (8, 5), (8, 6), (10, 5), (10, 6),
        (10, 7), (12, 6), (12, 7), (12, 8), (15, 7), (15, 8), (15, 9)
    ]
    random.shuffle(configurations)
    
    import time
    t_start_search = time.time()
    
    for num_frags, min_coarse_parts in configurations:
        random.shuffle(possible_start_edges)
        
        if coarse_edge_to_fine_bridges is None:
            continue
            
        for c_idx1, c_idx2 in possible_start_edges:
            # Try both orderings for symmetric bridges
            bridges = coarse_edge_to_fine_bridges.get((c_idx1, c_idx2))
            if not bridges:
                bridges = coarse_edge_to_fine_bridges.get((c_idx2, c_idx1))
            if not bridges:
                continue
            
            f1, f2 = random.choice(bridges)
            
            # BFS to expand connected fine partitions
            q_fine_indices, queue, visited = [f1, f2], [f1, f2], {f1, f2}
            while queue and len(q_fine_indices) < num_frags:
                current_fine_idx = queue.pop(0)
                current_c_idx = fine_to_coarse_map[current_fine_idx]
                coarse_neighbors_and_self = list(coarse_part_graph.neighbors(current_c_idx)) + [current_c_idx]
                potential_fine_neighbors = [fn for c_idx in coarse_neighbors_and_self for fn in coarse_to_fine_map.get(c_idx, [])]
                random.shuffle(potential_fine_neighbors)
                
                for neighbor_idx in potential_fine_neighbors:
                    if neighbor_idx not in visited and are_fine_partitions_connected(current_fine_idx, neighbor_idx, coarse_edge_to_fine_bridges, fine_to_coarse_map):
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)
                        q_fine_indices.append(neighbor_idx)
                        if len(q_fine_indices) >= num_frags:
                            break
                        
            if len(q_fine_indices) < num_frags:
                continue
            true_coarse_indices = {fine_to_coarse_map[f_idx] for f_idx in q_fine_indices}
            if len(true_coarse_indices) < min_coarse_parts:
                continue
            
            # Extract nodes from each fragment via random walk
            nodes_per_frag = max_nodes // num_frags
            all_query_nodes = []
            for fine_idx in q_fine_indices:
                # fine_graphs is now a dictionary
                local_nodes = _extract_fragment_fast_rw(fine_graphs[fine_idx], nodes_per_frag)
                if local_nodes is not None and local_nodes.max() < fine_part_nodes_map[fine_idx].size(0):
                    all_query_nodes.extend(fine_part_nodes_map[fine_idx][local_nodes].tolist())
            
            Gq, q_global_nodes = _finalize_query_from_nodes(original_data, adj_t, all_query_nodes, min_nodes)
            if Gq is None:
                continue
            
            # Stitch fine partitions into Gpos
            stitched_nodes = torch.cat([fine_part_nodes_map[idx] for idx in q_fine_indices])
            if len(stitched_nodes) > max_nodes * 5:
                stitched_nodes = _bounded_node_sample(stitched_nodes, max_nodes * 5, q_global_nodes)
            G_stitched = _extract_subgraph_from_adj(adj_t, stitched_nodes, original_data, preserve_nodes=q_global_nodes)
            
            duration = time.time() - t_start_search
            if duration > 5.0 and random.random() < 0.001:
                print(f"[PROFILE] multi-coarse match: {duration:.4f}s", file=sys.stderr)
            return Gq, G_stitched, list(true_coarse_indices), {
                'type': 'multi-coarse-opt',
                'time': duration,
                'query_global_ids': q_global_nodes.cpu().tolist(),
                'coverage_coarse_ids': list(true_coarse_indices),
                'query_fine_ids': list(q_fine_indices),
                'coverage_fine_ids': list(q_fine_indices),
            }
                    
    raise RuntimeError("Failed to generate multi-coarse-partition query.")

def generate_hierarchical_sample(original_data, adj_t, coarse_graphs, fine_graphs, 
                                  node_to_coarse_tensor, node_to_fine_tensor,
                                  fine_to_coarse_map, coarse_to_fine_map,
                                  coarse_edges_list, fine_part_nodes_map, coarse_part_nodes_map, 
                                  coarse_part_graph, # Added to support hard negative generation
                                  num_frags=4, q_size_min=20, q_size_max=120, 
                                  prob_k_hop=0.35, prob_single_part=0.15, prob_multi_coarse=0.30, 
                                  prob_random_walk=0.0, prob_degree_k_hop=0.0,
                                  max_gpos_nodes=4000, coarse_edge_to_fine_bridges=None,
                                  max_train_coarse_parts=20, coarse_graph_data_cache=None,
                                  hard_negative_source="graphs",
                                  query_target_sizes=None, query_size_jitter=5):
    """Generate (Gq, Gpos, G_coarse_pos) training sample using hierarchical sampling."""
    from torch_geometric.utils import k_hop_subgraph
    import time
    
    t0 = time.time()
    rand_choice = random.random()
    device = original_data.x.device
    Gq, Gpos, G_coarse_pos = None, None, None
    hard_negative_coarse_parts = []
    sample_type = "unknown"
    metadata = {'query_coarse_ids': [], 'query_fine_ids': [], 'coverage_fine_ids': []}

    def get_coarse_partition_graph(part_id):
        part_id = int(part_id)
        if coarse_graph_data_cache is not None:
            cached = coarse_graph_data_cache.get(part_id)
            if cached is not None:
                return cached
        graph = _extract_subgraph_from_adj(adj_t, coarse_part_nodes_map[part_id], original_data)
        if graph is not None and coarse_graph_data_cache is not None:
            coarse_graph_data_cache[part_id] = graph
        return graph

    def record_hard_negative_ids(hn_ids):
        if not hn_ids:
            return
        metadata.setdefault("hard_negative_coarse_ids", []).extend(int(hn_id) for hn_id in hn_ids)

    def build_context_from_query_nodes(query_nodes, sample_label, started_at):
        if query_nodes is None or len(query_nodes) < min_nodes_for_query:
            return None

        subset_coarse_ids = node_to_coarse_tensor[query_nodes]
        mask = subset_coarse_ids >= 0
        if mask.sum() == 0:
            return None

        unique_coarse_ids, counts = torch.unique(subset_coarse_ids[mask], return_counts=True)
        all_unique_coarse_ids = unique_coarse_ids
        all_unique_fine_ids = torch.empty(0, dtype=torch.long, device=query_nodes.device)
        if node_to_fine_tensor is not None:
            subset_fine_ids = node_to_fine_tensor[query_nodes]
            fine_mask = subset_fine_ids >= 0
            if fine_mask.any():
                all_unique_fine_ids = torch.unique(subset_fine_ids[fine_mask])

        k_partitions = max(1, max_train_coarse_parts)
        context_coarse_ids = unique_coarse_ids
        if len(context_coarse_ids) > k_partitions:
            _, top_indices = torch.topk(counts, k_partitions)
            context_coarse_ids = context_coarse_ids[top_indices]

        target_device = original_data.x.device
        pos_nodes_list = [
            coarse_part_nodes_map[int(cid.item())].to(target_device)
            for cid in context_coarse_ids
        ]
        if not pos_nodes_list:
            return None
        all_pos_nodes = torch.cat(pos_nodes_list).unique()
        if len(all_pos_nodes) > max_gpos_nodes:
            all_pos_nodes = _bounded_node_sample(all_pos_nodes, max_gpos_nodes, query_nodes)

        try:
            gpos = _extract_subgraph_from_adj(adj_t, all_pos_nodes, original_data, preserve_nodes=query_nodes)
            gq = _extract_subgraph_from_adj(adj_t, query_nodes, original_data)
            coarse_pos_nodes = torch.cat([
                coarse_part_nodes_map[int(cid.item())].to(target_device)
                for cid in context_coarse_ids
            ])
            if len(coarse_pos_nodes) > max_gpos_nodes:
                coarse_pos_nodes = _bounded_node_sample(coarse_pos_nodes, max_gpos_nodes, query_nodes)
            g_coarse_pos = _extract_subgraph_from_adj(
                adj_t, coarse_pos_nodes, original_data, preserve_nodes=query_nodes
            )
        except RuntimeError:
            return None

        hn_candidates = set()
        for cid in context_coarse_ids:
            cid_int = int(cid.item())
            if coarse_part_graph.has_node(cid_int):
                hn_candidates.update(coarse_part_graph.neighbors(cid_int))
        hn_candidates = list(hn_candidates - {int(cid.item()) for cid in context_coarse_ids})
        if hn_candidates:
            hn_ids = random.sample(hn_candidates, min(8, len(hn_candidates)))
            record_hard_negative_ids(hn_ids)
            if hard_negative_source == "graphs":
                for hn_id in hn_ids:
                    hn_graph = get_coarse_partition_graph(hn_id)
                    if hn_graph is not None:
                        hard_negative_coarse_parts.append(hn_graph)

        duration = time.time() - started_at
        if duration > 5.0 and random.random() < 0.001:
            print(
                f"[PROFILE] {sample_label}: {duration:.4f}s "
                f"(query_size:{len(query_nodes)}, pos_size:{len(all_pos_nodes)}, "
                f"parts:{len(context_coarse_ids)})",
                file=sys.stderr,
            )
        return gq, gpos, g_coarse_pos, {
            'type': sample_label,
            'time': duration,
            'query_global_ids': query_nodes.cpu().tolist(),
            'query_coarse_ids': [int(cid.item()) for cid in context_coarse_ids],
            'coverage_coarse_ids': [int(cid.item()) for cid in all_unique_coarse_ids],
            'query_fine_ids': [int(fid.item()) for fid in all_unique_fine_ids],
            'coverage_fine_ids': [int(fid.item()) for fid in all_unique_fine_ids],
            'coverage_target_count': int(len(all_unique_coarse_ids)),
            'context_target_count': int(len(context_coarse_ids)),
            'coverage_fine_target_count': int(len(all_unique_fine_ids)),
        }
    
    # Target exact semantic scales requested by user (with tiny variance to prevent overfitting to exact numbers)
    # Target exact semantic scales requested by user
    default_target_sizes = [20, 20, 20, 50, 100]  # 60% bias toward 20-node queries (matches eval)
    TARGET_Q_SIZES = list(query_target_sizes or default_target_sizes)
    # Filter to respect the caller's min/max range
    valid_sizes = [q for q in TARGET_Q_SIZES if q_size_min <= q <= q_size_max]
    base_q_size = random.choice(valid_sizes) if valid_sizes else random.randint(q_size_min, q_size_max)
    query_size_jitter = max(0, int(query_size_jitter))
    current_q_size = max(5, base_q_size + random.randint(-query_size_jitter, query_size_jitter))
    min_nodes_for_query = current_q_size - 10

    # -------------------------------------------------------------------------
    # STRATEGY 1: K-HOP SAMPLING
    # -------------------------------------------------------------------------
    if rand_choice < prob_k_hop:
        sample_type = "k-hop"
        t_0_khop = time.time()
        
        anchor = torch.randint(0, original_data.num_nodes, (1,), device=device).item()
        
        K_HOP = 3 # Hardcoded factor for strategy 1
        query_nodes = _sample_bounded_k_hop_nodes(
            adj_t, anchor, K_HOP, current_q_size, device=device
        )
        if query_nodes is None or len(query_nodes) < current_q_size:
            return None

        # Get coarse partitions overlapping with query
        subset_coarse_ids = node_to_coarse_tensor[query_nodes]
        mask = subset_coarse_ids >= 0
        if mask.sum() == 0:
            return None
            
        unique_coarse_ids, counts = torch.unique(subset_coarse_ids[mask], return_counts=True)
        all_unique_coarse_ids = unique_coarse_ids
        all_unique_fine_ids = torch.empty(0, dtype=torch.long, device=query_nodes.device)
        if node_to_fine_tensor is not None:
            subset_fine_ids = node_to_fine_tensor[query_nodes]
            fine_mask = subset_fine_ids >= 0
            if fine_mask.any():
                all_unique_fine_ids = torch.unique(subset_fine_ids[fine_mask])
        
        # Bound positive context for memory, but keep the full set as coverage targets.
        k_partitions = max(1, max_train_coarse_parts)
        if len(unique_coarse_ids) > k_partitions:
            _, top_indices = torch.topk(counts, k_partitions)
            unique_coarse_ids = unique_coarse_ids[top_indices]
        
        # Build Gpos from selected partitions
        target_device = original_data.x.device
        pos_nodes_list = [coarse_part_nodes_map[cid.item()].to(target_device) for cid in unique_coarse_ids]
        if not pos_nodes_list:
            return None
        all_pos_nodes = torch.cat(pos_nodes_list).unique()
        
        # Subsample NODES not partitions — preserves partition diversity while bounding size.
        # Shuffle so we don't systematically bias toward low-index nodes.
        if len(all_pos_nodes) > max_gpos_nodes:
            all_pos_nodes = _bounded_node_sample(all_pos_nodes, max_gpos_nodes, query_nodes)
        
        try:
            Gpos = _extract_subgraph_from_adj(adj_t, all_pos_nodes, original_data, preserve_nodes=query_nodes)
        except RuntimeError:
            return None
        
        Gq = _extract_subgraph_from_adj(adj_t, query_nodes, original_data)
        
        # Use all overlapping partitions as coarse context (consistent with coverage loss)
        all_coarse_pos_nodes = torch.cat([coarse_part_nodes_map[cid.item()].to(target_device) for cid in unique_coarse_ids])
        if len(all_coarse_pos_nodes) > max_gpos_nodes:
            all_coarse_pos_nodes = _bounded_node_sample(all_coarse_pos_nodes, max_gpos_nodes, query_nodes)
        G_coarse_pos = _extract_subgraph_from_adj(adj_t, all_coarse_pos_nodes, original_data, preserve_nodes=query_nodes)
        
        # Hard Negatives: Neighbors of ALL overlapping partitions
        hn_candidates = set()
        for cid in unique_coarse_ids:
            if coarse_part_graph.has_node(cid.item()):
                hn_candidates.update(coarse_part_graph.neighbors(cid.item()))
        hn_candidates = list(hn_candidates - {cid.item() for cid in unique_coarse_ids})
        if hn_candidates:
            hn_ids = random.sample(hn_candidates, min(8, len(hn_candidates)))
            record_hard_negative_ids(hn_ids)
            if hard_negative_source == "graphs":
                for hn_id in hn_ids:
                    hn_graph = get_coarse_partition_graph(hn_id)
                    if hn_graph is not None:
                        hard_negative_coarse_parts.append(hn_graph)
        
        dur_khop = time.time() - t_0_khop
        # Updated profile log
        if dur_khop > 5.0 and random.random() < 0.001:
            print(f"[PROFILE] k-hop({K_HOP}) total:{dur_khop:.4f}s (query_size:{len(query_nodes)}, pos_size:{len(all_pos_nodes)}, parts:{len(unique_coarse_ids)})", file=sys.stderr)
        metadata.update({'type': 'k-hop', 'time': dur_khop})
        metadata['query_coarse_ids'] = [cid.item() for cid in unique_coarse_ids]
        metadata['coverage_coarse_ids'] = [cid.item() for cid in all_unique_coarse_ids]
        metadata['query_fine_ids'] = [fid.item() for fid in all_unique_fine_ids]
        metadata['coverage_fine_ids'] = [fid.item() for fid in all_unique_fine_ids]
        metadata['coverage_target_count'] = len(metadata['coverage_coarse_ids'])
        metadata['context_target_count'] = len(metadata['query_coarse_ids'])
        metadata['coverage_fine_target_count'] = len(metadata['coverage_fine_ids'])

    # -------------------------------------------------------------------------
    # STRATEGY 2: SINGLE PARTITION SAMPLING
    # -------------------------------------------------------------------------
    elif rand_choice < prob_k_hop + prob_single_part:
        sample_type = "single-part"
        t_single = time.time()
        
        if not fine_graphs:
            return None
        fine_idx = random.choice(list(fine_to_coarse_map.keys()))
        Gpos = fine_graphs[fine_idx]
        
        if Gpos.num_nodes > max_gpos_nodes:
            return None
        
        q_nodes_local = _extract_fragment_fast_rw(Gpos, current_q_size)
        if q_nodes_local is None:
            return None
        
        pos_nodes_global = fine_part_nodes_map[fine_idx]
        q_nodes_global = pos_nodes_global[q_nodes_local]
        Gpos = _extract_subgraph_from_adj(adj_t, pos_nodes_global, original_data, preserve_nodes=q_nodes_global)
        Gq = _extract_subgraph_from_adj(adj_t, q_nodes_global, original_data)
        
        coarse_parent_idx = fine_to_coarse_map.get(fine_idx)
        if coarse_parent_idx is None:
            return None
        G_coarse_pos = _extract_subgraph_from_adj(adj_t, coarse_part_nodes_map[coarse_parent_idx], original_data, preserve_nodes=q_nodes_global)
        
        # Hard Negatives
        if coarse_part_graph.has_node(coarse_parent_idx):
            hn_candidates = list(coarse_part_graph.neighbors(coarse_parent_idx))
            if hn_candidates:
                hn_ids = random.sample(hn_candidates, min(8, len(hn_candidates)))
                record_hard_negative_ids(hn_ids)
                if hard_negative_source == "graphs":
                    for hn_id in hn_ids:
                        hn_graph = get_coarse_partition_graph(hn_id)
                        if hn_graph is not None:
                            hard_negative_coarse_parts.append(hn_graph)
        
        duration = time.time() - t_single
        if duration > 5.0 and random.random() < 0.001:
            print(f"[PROFILE] single-part: {duration:.4f}s", file=sys.stderr)
        metadata.update({'type': 'single-part', 'time': duration})
        metadata['query_coarse_ids'] = [coarse_parent_idx]
        metadata['coverage_coarse_ids'] = [coarse_parent_idx]
        metadata['query_fine_ids'] = [fine_idx]
        metadata['coverage_fine_ids'] = [fine_idx]
        
    # -------------------------------------------------------------------------
    # STRATEGY 3: MULTI-COARSE PARTITION SAMPLING
    # -------------------------------------------------------------------------
    elif rand_choice < prob_k_hop + prob_single_part + prob_multi_coarse:
        sample_type = "multi-coarse"
        t_multi = time.time()
        try:
            res = generate_multi_coarse_partition_query(
                original_data, adj_t, coarse_part_graph, fine_graphs, 
                fine_part_nodes_map, fine_to_coarse_map, coarse_to_fine_map, 
                coarse_edges_list, coarse_edge_to_fine_bridges=coarse_edge_to_fine_bridges,
                min_nodes=min_nodes_for_query, max_nodes=current_q_size
            )
            
            if res is None:
                return None
            Gq, Gpos, coarse_indices, meta_mc = res
            metadata.update(meta_mc)
            metadata['query_coarse_ids'] = list(coarse_indices)
            metadata['coverage_coarse_ids'] = list(meta_mc.get('coverage_coarse_ids', coarse_indices))
            metadata['query_fine_ids'] = list(meta_mc.get('query_fine_ids', []))
            metadata['coverage_fine_ids'] = list(meta_mc.get('coverage_fine_ids', metadata['query_fine_ids']))
            
            all_coarse_pos_nodes = torch.cat([coarse_part_nodes_map[c_idx] for c_idx in coarse_indices])
            q_global_ids = torch.tensor(meta_mc.get('query_global_ids', []), dtype=torch.long, device=all_coarse_pos_nodes.device)
            if len(all_coarse_pos_nodes) > max_gpos_nodes:
                all_coarse_pos_nodes = _bounded_node_sample(all_coarse_pos_nodes, max_gpos_nodes, q_global_ids)
            G_coarse_pos = _extract_subgraph_from_adj(adj_t, all_coarse_pos_nodes, original_data, preserve_nodes=q_global_ids)
            
            # Hard Negatives
            hn_candidates = set()
            for c_idx in coarse_indices:
                if coarse_part_graph.has_node(c_idx):
                    hn_candidates.update(list(coarse_part_graph.neighbors(c_idx)))
            hn_candidates = list(hn_candidates - set(coarse_indices))
            if hn_candidates:
                hn_ids = random.sample(hn_candidates, min(8, len(hn_candidates)))
                record_hard_negative_ids(hn_ids)
                if hard_negative_source == "graphs":
                    for hn_id in hn_ids:
                        hn_graph = get_coarse_partition_graph(hn_id)
                        if hn_graph is not None:
                            hard_negative_coarse_parts.append(hn_graph)
            
        except RuntimeError:
            return None
        
    # -------------------------------------------------------------------------
    # STRATEGY 4: FULL-GRAPH RANDOM-WALK SAMPLING
    # -------------------------------------------------------------------------
    elif rand_choice < prob_k_hop + prob_single_part + prob_multi_coarse + prob_random_walk:
        sample_type = "random-walk"
        t_walk = time.time()
        query_nodes = _sample_full_graph_random_walk_nodes(
            adj_t, current_q_size, device=device
        )
        built = build_context_from_query_nodes(query_nodes, sample_type, t_walk)
        if built is None:
            return None
        Gq, Gpos, G_coarse_pos, meta_rw = built
        metadata.update(meta_rw)

    # -------------------------------------------------------------------------
    # STRATEGY 5: DEGREE-BIASED K-HOP SAMPLING
    # -------------------------------------------------------------------------
    elif rand_choice < (
        prob_k_hop + prob_single_part + prob_multi_coarse
        + prob_random_walk + prob_degree_k_hop
    ):
        sample_type = "degree-k-hop"
        t_degree = time.time()
        query_nodes = _sample_degree_k_hop_nodes(
            adj_t, current_q_size, device=device
        )
        built = build_context_from_query_nodes(query_nodes, sample_type, t_degree)
        if built is None:
            return None
        Gq, Gpos, G_coarse_pos, meta_degree = built
        metadata.update(meta_degree)

    # -------------------------------------------------------------------------
    # STRATEGY 6: SIBLING-WALK SAMPLING
    # -------------------------------------------------------------------------
    else:
        sample_type = "sibling-walk"
        t_walk = time.time()
        if not fine_part_nodes_map or len(fine_part_nodes_map) < 2:
            return None
        
        Gpos = None
        source_part_indices = None
        
        # Retry loop to find connected sibling partitions
        for attempt in range(10):
            num_frags = random.randint(2, 5)
            start_fine_idx = random.choice(list(fine_part_nodes_map.keys()))
            coarse_parent_idx = fine_to_coarse_map.get(start_fine_idx)
            if coarse_parent_idx is None:
                continue
            
            siblings = coarse_to_fine_map.get(coarse_parent_idx, [])
            if len(siblings) < num_frags:
                continue

            potential_neighbors = [s for s in siblings if s != start_fine_idx]
            random.shuffle(potential_neighbors)
            current_cluster = [start_fine_idx]
            
            # Greedy expansion: add connected neighbors
            for candidate in potential_neighbors:
                for node in current_cluster:
                    if are_fine_partitions_connected(node, candidate, coarse_edge_to_fine_bridges, fine_to_coarse_map):
                        current_cluster.append(candidate)
                        break
                if len(current_cluster) >= num_frags:
                    break
            
            if len(current_cluster) >= num_frags:
                source_part_indices = set(current_cluster)
                pos_nodes = torch.cat([fine_part_nodes_map[i] for i in source_part_indices])
                if len(pos_nodes) <= max_gpos_nodes:
                    Gpos = _extract_subgraph_from_adj(adj_t, pos_nodes, original_data)
                    break

        if Gpos is None:
            return None
        
        # Extract query nodes from fragments
        nodes_per_frag = current_q_size // num_frags
        all_query_global_nodes = []
        for fine_idx in source_part_indices:
            # fine_graphs is now a dictionary
            local_indices = _extract_fragment_fast_rw(fine_graphs[fine_idx], nodes_per_frag)
            if local_indices is None:
                return None
            all_query_global_nodes.extend(
                fine_part_nodes_map[fine_idx][local_indices].tolist()
            )
            
        Gq, q_nodes_global = _finalize_query_from_nodes(original_data, adj_t, all_query_global_nodes, min_nodes=min_nodes_for_query)
        if Gq is None:
            return None
        G_coarse_pos = _extract_subgraph_from_adj(adj_t, coarse_part_nodes_map[coarse_parent_idx], original_data, preserve_nodes=q_nodes_global)
        
        # Hard Negatives
        if coarse_part_graph.has_node(coarse_parent_idx):
            hn_candidates = list(coarse_part_graph.neighbors(coarse_parent_idx))
            if hn_candidates:
                hn_ids = random.sample(hn_candidates, min(8, len(hn_candidates)))
                record_hard_negative_ids(hn_ids)
                if hard_negative_source == "graphs":
                    for hn_id in hn_ids:
                        hn_graph = get_coarse_partition_graph(hn_id)
                        if hn_graph is not None:
                            hard_negative_coarse_parts.append(hn_graph)
        
        duration = time.time() - t_walk
        if duration > 5.0 and random.random() < 0.001:
            print(f"[PROFILE] sibling-walk: {duration:.4f}s", file=sys.stderr)
        metadata.update({'type': 'sibling-walk', 'time': duration})
        metadata['query_coarse_ids'] = [coarse_parent_idx]
        metadata['coverage_coarse_ids'] = [coarse_parent_idx]
        metadata['query_fine_ids'] = list(source_part_indices)
        metadata['coverage_fine_ids'] = list(source_part_indices)
        
    # Final validation
    if Gq is None or Gpos is None or G_coarse_pos is None:
        return None
    if 'metadata' not in locals():
        metadata = {'type': sample_type, 'time': time.time() - t0}
    metadata['target_query_size'] = int(current_q_size)
    metadata['query_node_count'] = int(getattr(Gq, "num_nodes", 0))
    metadata['coverage_target_count'] = len(
        metadata.get('coverage_coarse_ids', metadata.get('query_coarse_ids', []))
    )
    metadata['context_target_count'] = len(metadata.get('query_coarse_ids', []))
    metadata['coverage_fine_target_count'] = len(
        metadata.get('coverage_fine_ids', metadata.get('query_fine_ids', []))
    )
    return Gq, Gpos, G_coarse_pos, metadata, hard_negative_coarse_parts

class JigsawDataset(Dataset):
    def __init__(self, original_data, adj_t, hierarchies, batch_size, steps_per_epoch, sample_kwargs=None):
        self.original_data = original_data
        self.adj_t = adj_t # Full GPU SparseTensor
        self.hierarchies = hierarchies
        self.sample_kwargs = sample_kwargs or {}
        
        # Optimize: Pre-convert node/partition maps to tensors for each hierarchy
        self.node_to_coarse_tensors = []
        self.node_to_fine_tensors = []
        for h_data in hierarchies:
            node_map_dict = h_data['node_to_coarse_map']
            # Create a tensor initialized with -1 or a valid default
            # Assuming nodes are 0..num_nodes-1
            # Using int32 should be enough for coarse IDs
            mapper = torch.full((original_data.num_nodes,), -1, dtype=torch.long, device=original_data.x.device)
            
            # This creation is one-time but might be slow if loop. 
            # Ideally we construct from keys/values tensors.
            # node_map_dict keys are global IDs.
            keys = torch.tensor(list(node_map_dict.keys()), dtype=torch.long)
            values = torch.tensor(list(node_map_dict.values()), dtype=torch.long)
            # Move to GPU for assignment
            keys = keys.to(original_data.x.device)
            values = values.to(original_data.x.device)
            mapper[keys] = values
            self.node_to_coarse_tensors.append(mapper)

            fine_mapper = torch.full((original_data.num_nodes,), -1, dtype=torch.long, device=original_data.x.device)
            fine_part_nodes_map = h_data.get('fine_part_nodes_map', {})
            for fid, nodes in fine_part_nodes_map.items():
                if nodes is None or nodes.numel() == 0:
                    continue
                fine_mapper[nodes.to(original_data.x.device)] = int(fid)
            self.node_to_fine_tensors.append(fine_mapper)

            # Pre-compute reverse map and edges for multi-coarse sampling optimization
            if 'precomputed_coarse_to_fine' not in h_data:
                c2f = defaultdict(list)
                f2c = h_data['fine_to_coarse_map']
                for f, c in f2c.items():
                    c2f[c].append(f)
                h_data['precomputed_coarse_to_fine'] = c2f
            
            if 'precomputed_coarse_edges' not in h_data:
                h_data['precomputed_coarse_edges'] = list(h_data['coarse_part_graph'].edges())
            
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
    
    def __len__(self):
        return self.steps_per_epoch * self.batch_size

    def generate_sample(self):
        # Everything happens in the same process, directly on GPU tensors
        # Loop until we get a valid sample (failed samples return None)
        while True:
            # Randomly select a hierarchy each time we retry
            h_idx = random.randint(0, len(self.hierarchies) - 1)
            h_data = self.hierarchies[h_idx]
            node_mapper = self.node_to_coarse_tensors[h_idx]
            fine_mapper = self.node_to_fine_tensors[h_idx]
            
            # Unpack hierarchy data
            try:
                sample = generate_hierarchical_sample(
                    self.original_data, self.adj_t, 
                    h_data['coarse_graphs'], h_data['fine_graphs'], 
                    node_mapper, # Passing Tensor instead of dict
                    fine_mapper,
                    h_data['fine_to_coarse_map'],
                    h_data['precomputed_coarse_to_fine'],
                    list(h_data['precomputed_coarse_edges']), # Pass a copy or list to shuffle inside
                    h_data['fine_part_nodes_map'], 
                    h_data['coarse_part_nodes_map'],
                    h_data['coarse_part_graph'], # Essential for hard negative generation
                    coarse_edge_to_fine_bridges=h_data.get('coarse_edge_to_fine_bridges'),
                    **self.sample_kwargs
                )
                if sample:
                    # Inject h_idx into metadata
                    sample[3]['hierarchy_idx'] = h_idx
                    return sample
            except Exception as e:
                # print(f"[WARN] Sample generation failed: {e}", file=sys.stderr)
                continue # Retry on error

    def __getitem__(self, idx):
        return self.generate_sample()

def jigsaw_collate_fn(batch_list):
    from torch_geometric.data import Batch
    gqs = []
    gpos = []
    gcs = []
    metadatas = []
    hns_list = []
    
    for b in batch_list:
        if b is None: continue 
        # tuple unpacking: (Gq, Gpos, G_coarse_pos, metadata, hns)
        if len(b) >= 5:
            item = b[0]
            if hasattr(item, 'part_id'): delattr(item, 'part_id')
            gqs.append(item)

            item = b[1]
            if hasattr(item, 'part_id'): delattr(item, 'part_id')
            gpos.append(item)

            item = b[2]
            if hasattr(item, 'part_id'): delattr(item, 'part_id')
            gcs.append(item)

            metadatas.append(b[3])
            hns_list.append(b[4])
        else:
            # Drop invalid batches
            continue

    # Batch.from_data_list crashes on empty list
    if len(gqs) == 0 or len(gpos) == 0 or len(gcs) == 0:
        return None
    
    try:
        return Batch.from_data_list(gqs), Batch.from_data_list(gpos), Batch.from_data_list(gcs), metadatas, hns_list
    except Exception as e:
        print(f"[WARN] Collate failed: {e}", flush=True)
        return None

# --- MODAL SETUP ---

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core packages
        "numpy<2.0", "networkx==3.2.1", "pymetis==2022.1", "torch==2.2.1",
        "torch_geometric==2.5.2", "torch-scatter==2.1.2", "torch-sparse==0.6.18",
        # OGB dependencies
        "ogb>=1.3.6", "torchdata==0.7.1", "pandas", "PyYAML", "pydantic", "tqdm",
        find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html",
    )
    # Set the library path so C++ extensions can find Torch's CUDA libs.
    .env({"LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/torch/lib"})
    .add_local_file("scripts/coverage_losses.py", remote_path="/root/coverage_losses.py")
)

app = modal.App("jigsaw-mag-training-full-graph", image=image)
cache_volume = modal.Volume.from_name("jigsaw-cache-vol", create_if_missing=True)

@app.function(
    image=image,
    gpu="l4",
    volumes={"/cache": cache_volume},
    timeout=86400,
    cpu=8.0,
    memory=65536, # Increased to 64GB for stability
)
def train(dataset_name, epochs, steps_per_epoch, batch_size, num_hierarchies=1,
          checkpoint_path="/cache/jigsaw_checkpoint.pth", fresh_start=False,
          run_name="default",
          gamma_partition=0.5, coverage_temperature=0.05,
          coverage_topk=0, coverage_topk_bucket_size=10,
          coverage_topk_weight=0.0, coverage_topk_margin=0.0,
          coverage_positive_aggregation="mean", coverage_cvar_fraction=0.25,
          coverage_smoothmax_temperature=0.1, max_live_positive_parts=0,
          gamma_fine_partition=0.0, fine_cache_refresh_steps=250,
          alpha=0.2, beta=0.0, prob_k_hop=0.35,
          prob_single_part=0.15, prob_multi_coarse=0.30,
          prob_random_walk=0.0, prob_degree_k_hop=0.0,
          hard_negative_source="graphs",
          max_gpos_nodes=4000, max_train_coarse_parts=20,
          query_target_sizes="20,20,20,50,100", query_size_jitter=5,
          cache_refresh_steps=20, cache_encode_batch_size=1,
          cache_partition_graphs=0, checkpoint_interval_epochs=2,
          resume_from_checkpoint="",
          learning_rate=1e-4, scheduler_type="plateau",
          min_learning_rate=1e-5, warmup_steps=100,
          plateau_patience=10, plateau_factor=0.5,
          cosine_t_max=0, resume_model_only=False,
          validation_queries=0, validation_interval=2, validation_seed=31415,
          validation_seeds="", validation_topks="20,50,100",
          early_stopping_patience=0, training_seed=42, disable_residual=False):
    epochs = int(epochs)
    steps_per_epoch = int(steps_per_epoch)
    batch_size = int(batch_size)
    num_hierarchies = int(num_hierarchies)
    gamma_partition = float(gamma_partition)
    gamma_fine_partition = float(gamma_fine_partition)
    coverage_temperature = float(coverage_temperature)
    coverage_topk = int(coverage_topk)
    coverage_topk_bucket_size = max(1, int(coverage_topk_bucket_size))
    coverage_topk_weight = float(coverage_topk_weight)
    coverage_topk_margin = float(coverage_topk_margin)
    coverage_positive_aggregation = str(coverage_positive_aggregation).strip().lower()
    coverage_cvar_fraction = float(coverage_cvar_fraction)
    coverage_smoothmax_temperature = float(coverage_smoothmax_temperature)
    max_live_positive_parts = int(max_live_positive_parts)
    fine_cache_refresh_steps = int(fine_cache_refresh_steps)
    alpha = float(alpha)
    beta = float(beta)
    prob_k_hop = float(prob_k_hop)
    prob_single_part = float(prob_single_part)
    prob_multi_coarse = float(prob_multi_coarse)
    prob_random_walk = float(prob_random_walk)
    prob_degree_k_hop = float(prob_degree_k_hop)
    hard_negative_source = str(hard_negative_source).strip().lower()
    if hard_negative_source not in {"graphs", "cache", "none"}:
        raise ValueError("hard_negative_source must be 'graphs', 'cache', or 'none'")
    max_gpos_nodes = int(max_gpos_nodes)
    max_train_coarse_parts = int(max_train_coarse_parts)
    query_target_sizes = [
        max(1, int(size.strip()))
        for size in str(query_target_sizes).split(",")
        if size.strip()
    ] or [20, 20, 20, 50, 100]
    query_size_jitter = max(0, int(query_size_jitter))
    cache_refresh_steps = int(cache_refresh_steps)
    cache_encode_batch_size = max(1, int(cache_encode_batch_size))
    cache_partition_graphs = bool(int(cache_partition_graphs))
    checkpoint_interval_epochs = max(1, int(checkpoint_interval_epochs))
    learning_rate = float(learning_rate)
    scheduler_type = str(scheduler_type).strip().lower()
    min_learning_rate = float(min_learning_rate)
    warmup_steps = int(warmup_steps)
    plateau_patience = int(plateau_patience)
    plateau_factor = float(plateau_factor)
    cosine_t_max = int(cosine_t_max)
    validation_queries = int(validation_queries)
    validation_interval = max(1, int(validation_interval))
    validation_seed = int(validation_seed)
    validation_seeds = [
        int(seed.strip())
        for seed in str(validation_seeds).split(",")
        if seed.strip()
    ] or [validation_seed]
    validation_topks = tuple(
        sorted(
            {
                max(1, int(topk.strip()))
                for topk in str(validation_topks).split(",")
                if topk.strip()
            }
        )
    ) or (20, 50, 100)
    early_stopping_patience = max(0, int(early_stopping_patience))
    training_seed = int(training_seed)
    if coverage_positive_aggregation not in {"mean", "cvar", "smoothmax"}:
        raise ValueError(
            "coverage_positive_aggregation must be mean, cvar, or smoothmax"
        )
    if not 0.0 < coverage_cvar_fraction <= 1.0:
        raise ValueError("coverage_cvar_fraction must be in (0, 1]")
    if isinstance(fresh_start, str):
        fresh_start = fresh_start.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(resume_model_only, str):
        resume_model_only = resume_model_only.strip().lower() in {"1", "true", "yes", "y"}

    # Force fresh start by using a new checkpoint name



    # --- REMOTE-ONLY IMPORTS and DEFINITIONS ---
    import itertools
    import random
    from collections import Counter, defaultdict
    import networkx as nx
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from torch.nn import Dropout, LeakyReLU, Linear, ReLU, Sequential, LayerNorm



    # --- REMOTE-ONLY IMPORTS and DEFINITIONS ---
    import itertools
    import random
    from collections import Counter, defaultdict
    import networkx as nx
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from torch.nn import Dropout, LeakyReLU, Linear, ReLU, Sequential, LayerNorm
    import sys
    import queue
    import threading
    import concurrent.futures

    random.seed(training_seed)
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    
    # CRITICAL STABILITY FIX: Remove thread restriction which clashes with torch_sparse OpenMP
    # Since num_workers=0, no forking occurs; restriction only caused C++ pool corruption
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # prevents HuggingFace tokenizer warnings
    
    # Configure stdout/stderr buffering
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    log_file = None

    class TeeStream:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    try:
        safe_run = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name)
        os.makedirs("/cache/logs", exist_ok=True)
        log_path = f"/cache/logs/train_{dataset_name}_{safe_run}.log"
        log_file = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = TeeStream(sys.__stdout__, log_file)
        sys.stderr = TeeStream(sys.__stderr__, log_file)
        print(f"[LOG] Training log will be written to {log_path}", flush=True)
    except Exception as e:
        print(f"[LOG] Could not initialize volume log file: {e}", flush=True)

    print("[CONFIG] Training controls:", flush=True)
    print(
        f"  gamma_partition={gamma_partition} gamma_fine_partition={gamma_fine_partition} "
        f"coverage_temperature={coverage_temperature}",
        flush=True,
    )
    print(
        f"  coverage_topk={coverage_topk} "
        f"coverage_topk_bucket_size={coverage_topk_bucket_size} "
        f"coverage_topk_weight={coverage_topk_weight} coverage_topk_margin={coverage_topk_margin}",
        flush=True,
    )
    print(
        f"  positive_aggregation={coverage_positive_aggregation} "
        f"cvar_fraction={coverage_cvar_fraction} "
        f"smoothmax_temperature={coverage_smoothmax_temperature} "
        f"max_live_positive_parts={max_live_positive_parts}",
        flush=True,
    )
    print(f"  fine_cache_refresh_steps={fine_cache_refresh_steps}", flush=True)
    print(f"  alpha={alpha} beta={beta}", flush=True)
    print(f"  hard_negative_source={hard_negative_source}", flush=True)
    print(
        f"  learning_rate={learning_rate:.2e} scheduler={scheduler_type} "
        f"min_lr={min_learning_rate:.2e} warmup_steps={warmup_steps}",
        flush=True,
    )
    sibling_prob = max(
        0.0,
        1.0
        - prob_k_hop
        - prob_single_part
        - prob_multi_coarse
        - prob_random_walk
        - prob_degree_k_hop,
    )
    print(
        f"  sample_mix: k_hop={prob_k_hop} single={prob_single_part} "
        f"multi_coarse={prob_multi_coarse} random_walk={prob_random_walk} "
        f"degree_k_hop={prob_degree_k_hop} sibling={sibling_prob:.2f}",
        flush=True,
    )
    print(
        f"  query_target_sizes={query_target_sizes} "
        f"query_size_jitter={query_size_jitter}",
        flush=True,
    )
    print(f"  max_gpos_nodes={max_gpos_nodes} max_train_coarse_parts={max_train_coarse_parts}", flush=True)
    print(
        f"  cache_refresh_steps={cache_refresh_steps} "
        f"cache_encode_batch_size={cache_encode_batch_size} "
        f"cache_partition_graphs={cache_partition_graphs} "
        f"checkpoint_interval_epochs={checkpoint_interval_epochs}",
        flush=True,
    )
    print(f"  resume_model_only={resume_model_only}", flush=True)
    print(
        f"  validation_queries_per_seed={validation_queries} "
        f"validation_interval={validation_interval} validation_seeds={validation_seeds} "
        f"validation_topks={validation_topks} "
        f"early_stopping_patience={early_stopping_patience} training_seed={training_seed}",
        flush=True,
    )


    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.data import Batch, Data, HeteroData
    from torch_geometric.nn import GINConv, global_mean_pool, GATConv, global_max_pool, global_add_pool
    from torch_sparse import SparseTensor
    from torch_geometric.datasets import CoraFull
    from torch_geometric.datasets import Planetoid
    from torch_geometric.datasets import Coauthor
    from torch_geometric.datasets import Flickr
    from torch_geometric.datasets import Yelp


    def run_pymetis_in_subprocess(n_parts, xadj, adjncy):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_partitioner_target, args=(q, n_parts, xadj, adjncy))
        p.start()
        try:
            # 10 minute timeout for partitioning
            result = q.get(timeout=600)
        except multiprocessing.queues.Empty:
            p.kill()
            raise RuntimeError(f"METIS partitioning timed out after 600s. Subprocess stuck?")
        p.join()
        if result is None: raise RuntimeError("Partitioning failed in subprocess.")
        return None, result

    # --- MAG CONVERSION & FEATURE AUGMENTATION HELPERS ---
    def convert_hetero_to_homo(hetero_data: "HeteroData") -> Data:
        """
        Convert OGBN-MAG HeteroData -> homogeneous Data.

        Paper nodes keep their OGB features. Other node types do not have raw
        OGB features, so append node-type one-hot features and per-relation
        degree features to avoid collapsing them to identical zero vectors.
        """
        print("  - Converting heterogeneous graph to homogeneous...")
        node_types = list(hetero_data.num_nodes_dict.keys())
        node_offset, total_nodes = {}, 0
        for nt in node_types:
            node_offset[nt] = total_nodes
            total_nodes += hetero_data.num_nodes_dict[nt]

        node_type_ids = torch.zeros(total_nodes, dtype=torch.long)
        type_features = torch.zeros(total_nodes, len(node_types), dtype=torch.float)
        for i, nt in enumerate(node_types):
            s, e = node_offset[nt], node_offset[nt] + hetero_data.num_nodes_dict[nt]
            node_type_ids[s:e] = i
            type_features[s:e, i] = 1.0

        if "paper" in hetero_data.x_dict:
            feat_dim = hetero_data.x_dict["paper"].size(1)
            base_x = torch.zeros(total_nodes, feat_dim, dtype=torch.float)
            p_start, p_end = node_offset["paper"], node_offset["paper"] + hetero_data.num_nodes_dict["paper"]
            base_x[p_start:p_end] = hetero_data.x_dict["paper"].float()
        else:
            base_x = torch.zeros(total_nodes, 128, dtype=torch.float)

        all_ei = []
        all_edge_type = []
        edge_types = list(hetero_data.edge_index_dict.keys())
        rel_degree = torch.zeros(total_nodes, max(1, 2 * len(edge_types)), dtype=torch.float)
        for rel_id, (edge_key, ei) in enumerate(hetero_data.edge_index_dict.items()):
            src_t, rel, dst_t = edge_key
            gei = ei.clone()
            gei[0] += node_offset[src_t]
            gei[1] += node_offset[dst_t]
            all_ei.append(gei)
            all_edge_type.append(torch.full((gei.size(1),), rel_id, dtype=torch.long))

            ones = torch.ones(gei.size(1), dtype=torch.float)
            rel_degree[:, 2 * rel_id].index_add_(0, gei[0], ones)
            rel_degree[:, 2 * rel_id + 1].index_add_(0, gei[1], ones)

        edge_index = torch.cat(all_ei, dim=1) if all_ei else torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.cat(all_edge_type, dim=0) if all_edge_type else torch.empty((0,), dtype=torch.long)
        rel_degree = torch.log1p(rel_degree)
        rel_degree = rel_degree / rel_degree.clamp_min(1.0).amax(dim=0, keepdim=True)
        x = torch.cat([base_x, type_features, rel_degree], dim=1).contiguous()

        homo = Data(x=x, edge_index=edge_index, num_nodes=total_nodes)
        homo.node_type = node_type_ids
        homo.node_types = node_types
        homo.node_offset = node_offset
        homo.edge_type = edge_type
        homo.edge_types = edge_types
        homo.feature_schema = "mag_type_rel_v1"
        homo.global_id = torch.arange(total_nodes, dtype=torch.long)
        paper_y = None
        if hasattr(hetero_data, "y_dict") and "paper" in hetero_data.y_dict:
            paper_y = hetero_data.y_dict["paper"]
        elif hasattr(hetero_data, "node_stores"):
            paper_store = next(
                (
                    store
                    for store in hetero_data.node_stores
                    if getattr(store, "_key", None) == "paper"
                ),
                None,
            )
            if paper_store is not None and hasattr(paper_store, "y"):
                paper_y = paper_store.y
        if paper_y is not None:
            y = torch.full((total_nodes,), -1, dtype=paper_y.dtype)
            p_start = node_offset["paper"]
            p_end = p_start + hetero_data.num_nodes_dict["paper"]
            y[p_start:p_end] = paper_y.view(-1)
            homo.y = y
        print(
            f"    - Converted to homogeneous: {homo.num_nodes} nodes, "
            f"{homo.edge_index.size(1)} edges, {homo.x.size(1)} features "
            f"(paper + type + relation-degree)",
            flush=True,
        )
        return homo

    # --- NodeFeatureAugmentor removed per user request ---

    def make_undirected_fast(edge_index, num_nodes):
        # This part runs on CPU initially or we can move edge_index to GPU first
        # Since we are immediately moving to GPU after, let's keep this as is for robust conversion
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to_symmetric()
        row, col, _ = adj.coo()
        return torch.stack([row, col], dim=0)

    # --- MODEL ARCHITECTURE ---
    class ImprovedSubgraphEncoder(torch.nn.Module):
        def __init__(self, in_neurons, hidden_neurons, output_neurons, dropout=0.1, use_residual=True):
            super().__init__()
            self.use_residual = use_residual
            self.dropout = dropout

            nn1 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
            self.conv1 = GINConv(nn1)
            nn2 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
            self.conv2 = GINConv(nn2)
            nn3 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
            self.conv3 = GINConv(nn3)
            nn4 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
            self.conv4 = GINConv(nn4)
            nn5 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
            self.conv5 = GINConv(nn5)
            nn6 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
            self.conv6 = GINConv(nn6)

            self.ln1 = LayerNorm(hidden_neurons); self.ln2 = LayerNorm(hidden_neurons); self.ln3 = LayerNorm(hidden_neurons)
            self.ln4 = LayerNorm(hidden_neurons); self.ln5 = LayerNorm(hidden_neurons); self.ln6 = LayerNorm(hidden_neurons)
            self.input_proj = Linear(in_neurons, hidden_neurons)
            self.use_multi_pool = True; readout_dim = hidden_neurons * 6 * 3

            self.readout_proj = Sequential(
                Linear(readout_dim, hidden_neurons * 2), ReLU(), Dropout(dropout),
                Linear(hidden_neurons * 2, hidden_neurons), ReLU(), Dropout(dropout),
                Linear(hidden_neurons, output_neurons)
            )
            self.readout_skip = Linear(readout_dim, output_neurons)

        def forward(self, x, edge_index, batch):
            layer_outputs = []
            feat = x.x if hasattr(x, 'x') else x

            # Project to hidden space immediately (Massive memory saving for activations)
            feat = F.relu(self.input_proj(feat))
            x_res = feat 

            h1 = F.relu(self.ln1(self.conv1(feat, edge_index) + (x_res if self.use_residual else 0)))
            layer_outputs.append(h1)
            h2 = F.relu(self.ln2(self.conv2(h1, edge_index) + (h1 if self.use_residual else 0)))
            layer_outputs.append(h2)
            h3 = F.relu(self.ln3(self.conv3(h2, edge_index) + (h2 if self.use_residual else 0)))
            layer_outputs.append(h3)
            h4 = F.relu(self.ln4(self.conv4(h3, edge_index) + (h3 if self.use_residual else 0)))
            layer_outputs.append(h4)
            h5 = F.relu(self.ln5(self.conv5(h4, edge_index) + (h4 if self.use_residual else 0)))
            layer_outputs.append(h5)
            h6 = F.relu(self.ln6(self.conv6(h5, edge_index) + (h5 if self.use_residual else 0)))
            layer_outputs.append(h6)

            pooled_representations = []
            for layer_out in layer_outputs:
                pooled_representations.extend([global_mean_pool(layer_out, batch), global_max_pool(layer_out, batch), global_add_pool(layer_out, batch)])
            h_final = torch.cat(pooled_representations, dim=1)
            graph_emb = F.normalize(self.readout_proj(h_final) + self.readout_skip(h_final), dim=1)
            
            # Return both graph-level embedding and final node-level embeddings (h6)
            node_emb = F.normalize(h6, dim=1)
            return graph_emb, node_emb

    # --- HIERARCHICAL LOSS WITH NODE ALIGNMENT & HARD NEGATIVES ---
    def info_nce_loss(zq, z_fine, z_coarse, hard_negatives=None, temperature=0.1):
        # zq: (B, D), positives: (B, D)
        # Positives
        pos_sim = torch.sum(zq * z_fine, dim=1, keepdim=True) / temperature
        
        # In-batch Negatives
        neg_sim = torch.matmul(zq, z_fine.T) / temperature
        # Remove self-similarity from negatives
        mask = torch.eye(len(zq), device=zq.device, dtype=torch.bool)
        neg_sim[mask] = -float('inf') 
        
        all_sims = [pos_sim, neg_sim]
        
        # Explicit Hard Negatives
        if hard_negatives is not None and len(hard_negatives) > 0:
            # hard_negatives: (B, num_hard_neg, D)
            B, num_hn, D = hard_negatives.shape
            # q_exp: (B, 1, D)
            q_exp = zq.unsqueeze(1)
            # hn_sim: (B, num_hard_neg)
            hn_sim = torch.sum(q_exp * hard_negatives, dim=2) / temperature
            all_sims.append(hn_sim)
            
        logits = torch.cat(all_sims, dim=1)
        # Label is always 0 because pos_sim is at index 0 
        labels = torch.zeros(len(zq), dtype=torch.long, device=zq.device)
        return F.cross_entropy(logits, labels)

    def node_alignment_loss(q_nodes, q_batch, p_nodes, p_batch, temperature=0.1):
        """Aligns node embeddings based on global_id using fast GPU scatter/gather.
        Uses first-occurrence deduplication to preserve neighborhood context signal.
        """
        if not hasattr(q_batch, 'global_id') or not hasattr(p_batch, 'global_id'):
            return 0.0

        q_gids = q_batch.global_id
        p_gids = p_batch.global_id
        
        # Deduplicate Q: Take first occurrence of each global_id in the batch
        q_unique_gids, q_inv = torch.unique(q_gids, return_inverse=True)
        # Fast way to get first index: scatter into a buffer of decreasing values
        rev_idx_q = len(q_gids) - 1 - torch.arange(len(q_gids), device=q_gids.device)
        first_q_indices = torch.zeros(len(q_unique_gids), dtype=torch.long, device=q_gids.device)
        first_q_indices.scatter_(0, q_inv, rev_idx_q)
        first_q_indices = len(q_gids) - 1 - first_q_indices
        q_nodes_unique = q_nodes[first_q_indices]
        
        # Deduplicate P: Take first occurrence of each global_id in the batch
        p_unique_gids, p_inv = torch.unique(p_gids, return_inverse=True)
        rev_idx_p = len(p_gids) - 1 - torch.arange(len(p_gids), device=p_gids.device)
        first_p_indices = torch.zeros(len(p_unique_gids), dtype=torch.long, device=p_gids.device)
        first_p_indices.scatter_(0, p_inv, rev_idx_p)
        first_p_indices = len(p_gids) - 1 - first_p_indices
        p_nodes_unique = p_nodes[first_p_indices]

        # Build lookup table on GPU: gid -> index in p_nodes_unique
        p_max_gid = p_unique_gids.max().item() + 1
        q_max_gid = q_unique_gids.max().item() + 1
        max_gid = max(p_max_gid, q_max_gid)
        
        p_lookup = torch.full((max_gid,), -1, dtype=torch.long, device=p_gids.device)
        p_lookup[p_unique_gids] = torch.arange(len(p_unique_gids), device=p_gids.device)
        
        p_indices = p_lookup[q_unique_gids]
        match_mask = p_indices >= 0
        
        if match_mask.sum() < 2:
            return 0.0
        
        q_final_feats = q_nodes_unique[match_mask]
        p_final_feats = p_nodes_unique[p_indices[match_mask]]
        
        logits = torch.matmul(q_final_feats, p_final_feats.T) / temperature
        labels = torch.arange(len(q_final_feats), device=q_final_feats.device)
        return F.cross_entropy(logits, labels)

    def hierarchical_info_nce_loss(zq, z_fine, z_coarse, q_node_emb, p_node_emb, q_batch, p_batch, hard_negatives=None, temperature=0.1, alpha=0.2, beta=0.0):
        loss_fine = info_nce_loss(zq, z_fine, z_coarse, temperature=temperature) # Simplified call
        loss_coarse = info_nce_loss(zq, z_coarse, z_coarse, hard_negatives=hard_negatives, temperature=temperature)
        # Node alignment loss disabled (beta=0.0) — node embeddings unused at inference,
        # gradient capacity fully available for retrieval-focused losses.
        loss_node = 0.0 if beta == 0.0 else node_alignment_loss(q_node_emb, q_batch, p_node_emb, p_batch, temperature=temperature)
        
        return (alpha * loss_fine) + ((1 - alpha - beta) * loss_coarse) + (beta * loss_node)

    from coverage_losses import partition_coverage_loss

    # --- DATA PARTITIONING AND HIERARCHY HELPERS ---
    def make_partitions(dataset, num_parts, keep_features=True):
        from torch_geometric.utils import subgraph
        from torch_geometric.data import Data
        
        # --- SANITY CHECKS ---
        if dataset.num_nodes == 0: return [], {}
        # Ensure consistency
        if dataset.x is not None and dataset.x.size(0) != dataset.num_nodes:
             raise RuntimeError(f"Dataset x size {dataset.x.size(0)} != num_nodes {dataset.num_nodes}")
        
        # Check edge index bounds (expensive but necessary for debugging this crash)
        # We are already moving to CPU below, so we can check there
        
        if dataset.num_nodes < num_parts: num_parts = dataset.num_nodes
        if num_parts <= 1: 
            d = Data(edge_index=dataset.edge_index, num_nodes=dataset.num_nodes)
            if keep_features:
                 d.x = dataset.x
                 d.y = dataset.y
            d.part_id = 0
            return [d], {0: torch.arange(dataset.num_nodes, device=dataset.edge_index.device)}
        
        # Partitioning needs to happen on CPU (pymetis requirement usually)
        # Validate edge index bounds on GPU (fast fail)
        if dataset.edge_index.numel() > 0:
             max_idx = dataset.edge_index.max().item()
             if max_idx >= dataset.num_nodes:
                  raise RuntimeError(f"Edge index max {max_idx} >= num_nodes {dataset.num_nodes}")

        edge_index_cpu = dataset.edge_index.cpu()
        if edge_index_cpu.numel() > 0 and edge_index_cpu.max() >= dataset.num_nodes:
             raise RuntimeError(f"Edge index contains indices >= num_nodes ({dataset.num_nodes}). Max: {edge_index_cpu.max()}")

        adj = SparseTensor.from_edge_index(edge_index_cpu, sparse_sizes=(dataset.num_nodes, dataset.num_nodes))
        xadj_t, adjncy_t, _ = adj.csr(); xadj, adjncy = xadj_t.tolist(), adjncy_t.tolist()
        
        _, membership = run_pymetis_in_subprocess(num_parts, xadj=xadj, adjncy=adjncy)
        
        part_graphs, part_nodes_map = [], {}
        for part_id in range(num_parts):
            node_indices = [i for i, p in enumerate(membership) if p == part_id]
            if node_indices:
                # Sanity check indices
                if max(node_indices) >= dataset.num_nodes:
                     raise RuntimeError(f"Partition {part_id} has indices >= num_nodes ({dataset.num_nodes})")
                
                nodes_tensor = torch.tensor(node_indices, dtype=torch.long, device=dataset.edge_index.device)
                part_nodes_map[part_id] = nodes_tensor
                
                # Manual Data construction to avoid implicit subgraph issues and ensure correct relabeling
                if dataset.edge_index.is_cuda:
                    torch.cuda.synchronize()
                relabeled_edge_index, _ = subgraph(nodes_tensor, dataset.edge_index, relabel_nodes=True, num_nodes=dataset.num_nodes)
                if dataset.edge_index.is_cuda:
                    torch.cuda.synchronize()
                

                # Verify relabeled edge index
                if relabeled_edge_index.numel() > 0 and relabeled_edge_index.max() >= len(node_indices):
                     raise RuntimeError(f"Relabeled edge index OOB: max {relabeled_edge_index.max()} >= num_sub_nodes {len(node_indices)}")
                
                part_data = Data(edge_index=relabeled_edge_index, num_nodes=len(nodes_tensor))
                part_data.part_id = part_id
                
                
                # Copy attributes manually to be safe
                if keep_features:
                    if dataset.x is not None:
                        part_data.x = dataset.x[nodes_tensor]
                    if dataset.y is not None:
                        part_data.y = dataset.y[nodes_tensor]
                    
                    # Copy masks if present
                    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
                        if hasattr(dataset, mask_name):
                            setattr(part_data, mask_name, getattr(dataset, mask_name)[nodes_tensor])
                    
                    # Copy node_type if present (essential for heterogeneous-to-homogeneous graphs)
                    if hasattr(dataset, 'node_type') and dataset.node_type is not None:
                        part_data.node_type = dataset.node_type[nodes_tensor]
                
                # Copy global metadata that doesn't need slicing
                for global_attr in ['node_types', 'node_offset', 'edge_types', 'edge_offset', 'feature_schema']:
                    if hasattr(dataset, global_attr):
                        setattr(part_data, global_attr, getattr(dataset, global_attr))

                if hasattr(dataset, 'global_id') and dataset.global_id is not None:
                    part_data.global_id = dataset.global_id[nodes_tensor]
                else:
                    # If global_id is missing, use the indices into the current dataset (which might be global)
                    part_data.global_id = nodes_tensor
                
                # OPTIMIZATION: Cache adj_t for fast random walks later
                if part_data.edge_index.numel() > 0:
                     try:
                         # Ensure we are on the correct device
                         part_data.adj_t = SparseTensor(row=part_data.edge_index[0], col=part_data.edge_index[1], 
                                                        sparse_sizes=(part_data.num_nodes, part_data.num_nodes))
                         part_data.adj_t.csr() # Pre-compute CSR
                     except Exception:
                         part_data.adj_t = None
                
                part_graphs.append(part_data)
            else:
                part_graphs.append(None)
        return part_graphs, part_nodes_map

    def build_single_hierarchy(data, num_coarse, num_fine):
        print(f"\n  * Building hierarchy with {num_coarse} coarse partitions...")
        # OPTIMIZATION: Do not keep features in hierarchy graphs to save RAM
        coarse_graphs, coarse_part_nodes_map = make_partitions(data, num_coarse, keep_features=False)
        
        # Move map to CPU for networkx graph construction, or keep as is.
        # Constructing coarse_part_graph (networkx) happens on CPU.
        node_to_coarse_map = {node_idx.item(): coarse_id for coarse_id, nodes in coarse_part_nodes_map.items() for node_idx in nodes}
        coarse_part_graph = nx.Graph()
        
        # This loop over edges is SLOW on CPU if done node-by-node in Python for 20M edges.
        # But coarse graph has fewer edges. 
        # Wait, iterating over ALL data.edge_index is required to find coarse edges.
        # Optimization: Map edges to coarse IDs using tensors, then distinct.
        
        print("    - Constructing coarse graph efficiently...", flush=True)
        # Vectorized coarse edge construction
        src, dst = data.edge_index
        # We need a tensor map from node_id -> coarse_id
        # coarse_ids tensor
        coarse_ids = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device) - 1
        for cid, nodes in coarse_part_nodes_map.items():
            coarse_ids[nodes] = cid
            
        c_src = coarse_ids[src]
        c_dst = coarse_ids[dst]
        
        # Filter inter-partition edges
        mask = (c_src != c_dst) & (c_src != -1) & (c_dst != -1)
        c_edges = torch.stack([c_src[mask], c_dst[mask]], dim=1)
        
        # Unique edges
        c_edges = torch.unique(c_edges, dim=0).cpu().numpy()
        coarse_part_graph.add_edges_from(c_edges)
        
        fine_graphs_dict, fine_part_nodes_map, fine_to_coarse_map = {}, {}, {}
        fine_global_idx = 0
        iterator = tqdm(enumerate(coarse_graphs), total=len(coarse_graphs), desc="    - Creating fine partitions", unit="coarse_part", ncols=100, mininterval=30.0)
        for coarse_list_idx, coarse_graph in iterator:
            if coarse_graph is None:
                continue
            # Fix for alignment: use original part_id if available
            coarse_idx = getattr(coarse_graph, 'part_id', coarse_list_idx)
            
            if coarse_idx not in coarse_part_nodes_map: continue
            global_nodes_of_this_coarse_part = coarse_part_nodes_map[coarse_idx]
            if coarse_graph.num_nodes < (num_fine * 2) or coarse_graph.num_edges == 0:
                finer_partitions, finer_nodes_map_local = [coarse_graph], {0: torch.arange(coarse_graph.num_nodes, device=data.x.device)}
            else: 
                # print(f"DEBUG: Processing coarse_idx {coarse_idx} with {coarse_graph.num_nodes} nodes, {coarse_graph.num_edges} edges", flush=True)
                finer_partitions, finer_nodes_map_local = make_partitions(coarse_graph, num_fine, keep_features=False)
            for fine_local_idx, fine_part in enumerate(finer_partitions):
                if fine_part is None or fine_local_idx not in finer_nodes_map_local:
                    continue
                local_indices_in_coarse = finer_nodes_map_local[fine_local_idx]
                global_indices_for_fine = global_nodes_of_this_coarse_part[local_indices_in_coarse]
                if global_indices_for_fine.numel() == 0 or fine_part.num_nodes == 0:
                    continue
                if fine_part.num_edges > 0:
                    # PRE-COMPUTE ADJ_T TO PREVENT RUNTIME SEGFAULTS
                    # Creating SparseTensor on-the-fly in training loop causes memory instability on CPU
                    from torch_sparse import SparseTensor
                    fine_part.adj_t = SparseTensor(
                        row=fine_part.edge_index[0], 
                        col=fine_part.edge_index[1], 
                        sparse_sizes=(fine_part.num_nodes, fine_part.num_nodes)
                    )
                    fine_part.adj_t.csr() # Pre-calculate CSR pointers for random walk

                fine_graphs_dict[fine_global_idx] = fine_part
                fine_part_nodes_map[fine_global_idx] = global_indices_for_fine
                fine_to_coarse_map[fine_global_idx] = coarse_idx
                fine_global_idx += 1
        # Pre-compute coarse_edge -> valid fine bridges
        # This requires mapping fine partitions back to nodes
        # 'fine_part_nodes_map' has global indices for each fine partition.
        # We need a global 'fine_id' tensor.
        fine_ids = torch.full((data.num_nodes,), -1, dtype=torch.long, device=data.x.device)
        for fid, nodes in fine_part_nodes_map.items():
            fine_ids[nodes] = fid
        
        # We also need coarse IDs (already computed as coarse_ids in previous block, but let's assume it might not serve purely or we need to ensure consistency)
        # Re-using 'coarse_ids' from previous block (it was a local variable, so we might need to re-compute or it's gone if out of scope? It is in scope).
        
        # Check if coarse_ids is still available?
        # Python scoping: variables in loops leak, but coarse_ids was defined at 805. It should be available.
        
        f_src = fine_ids[src]
        f_dst = fine_ids[dst]
        
        # Filter edges where fine partitions differ (potential bridges)
        # And ensure valid fine mapping
        bridge_mask = (f_src != f_dst) & (f_src != -1) & (f_dst != -1)
        
        # Filter further: only edges between DIFFERENT coarse partitions
        # (We only optimize multi-coarse sampling between coarse neighbors)
        bridge_mask = bridge_mask & (c_src != c_dst) & (c_src != -1) & (c_dst != -1)
        # Note: c_src/c_dst were defined in previous block.
        
        # Extract bridge pairs
        b_c_src = c_src[bridge_mask]
        b_c_dst = c_dst[bridge_mask]
        b_f_src = f_src[bridge_mask]
        b_f_dst = f_dst[bridge_mask]
        
        # Stack into (N, 4) tensor: [c1, c2, f1, f2]
        bridges_tensor = torch.stack([b_c_src, b_c_dst, b_f_src, b_f_dst], dim=1)
        
        # Unique bridges
        bridges_tensor = torch.unique(bridges_tensor, dim=0)
        
        # Move to CPU to build dictionary
        bridges_np = bridges_tensor.cpu().numpy()
        
        coarse_edge_to_fine_bridges = defaultdict(list)
        for r in bridges_np:
            c1, c2, f1, f2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            # Add symmetric entries because we shuffle edges and might query (c1,c2) or (c2,c1)
            coarse_edge_to_fine_bridges[(c1, c2)].append((f1, f2))
            coarse_edge_to_fine_bridges[(c2, c1)].append((f2, f1))
             
        # print(f"    - Pre-computed {len(bridges_np)} fine bridges", flush=True)

        return {
            'coarse_graphs': coarse_graphs,
            'fine_graphs': fine_graphs_dict,
            'node_to_coarse_map': node_to_coarse_map,
            'fine_to_coarse_map': fine_to_coarse_map,
            'fine_part_nodes_map': fine_part_nodes_map,
            'coarse_part_graph': coarse_part_graph,
            'coarse_part_nodes_map': coarse_part_nodes_map,
            'coarse_edge_to_fine_bridges': dict(coarse_edge_to_fine_bridges)
        }

    def build_multiple_hierarchies_target(data, dataset_name, n):
        print(f"[SETUP] Building {n} hierarchies for {dataset_name} for Jigsaw training...")
        
        # Hardcoded hierarchy targets based on dataset
        if dataset_name == 'arxiv':
            target_coarse = 200
            target_fine = 5
        elif dataset_name == 'cora':
            target_coarse = 20
            target_fine = 5
        elif dataset_name == 'mag':
            target_coarse = 2000
            target_fine = 5
        elif dataset_name == 'pubmed':
            target_coarse = 20
            target_fine = 5
        elif dataset_name == 'yelp':
            target_coarse = 700
            target_fine = 5
        elif dataset_name == 'physics':
            target_coarse = 35
            target_fine = 5
        elif dataset_name == 'citeseer':
            target_coarse = 10
            target_fine = 5
        elif dataset_name == 'flickr':
            target_coarse = 100
            target_fine = 5
        else:
            target_coarse = 50
            target_fine = 5
            
        print(f"  - Target configuration: Coarse={target_coarse}, Fine={target_fine}")
        
        all_hierarchies = []
        for i in range(n):
            print(f"    - Building hierarchy {i+1}/{n}...")
            hierarchy_data = build_single_hierarchy(data, target_coarse, target_fine)
            all_hierarchies.append(hierarchy_data)
            
        return all_hierarchies

    # --- CORE TRAINING LOGIC ---
    device = torch.device("cuda"); print(f"[REMOTE INFO] Using device: {device}", flush=True)
    
    # Dataset Loading Logic
    if dataset_name == 'mag':
        print("[REMOTE INFO] Loading OGBN-MAG (heterogeneous)...", flush=True)
        dataset = PygNodePropPredDataset(name="ogbn-mag", root="/tmp/ogbn_mag_data")
        data = convert_hetero_to_homo(dataset[0])
    elif dataset_name == 'cora':
        print("[REMOTE INFO] Loading CoraFull...", flush=True)
        dataset = CoraFull(root="/tmp/Cora")
        data = dataset[0]
        # Standardize attributes
        if not hasattr(data, 'node_types'):
            data.node_types = ['paper'] 
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
    elif dataset_name == 'arxiv':
        print("[REMOTE INFO] Loading OGBN-Arxiv...", flush=True)
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="/tmp/ogbn_arxiv")
        data = dataset[0]
        if not hasattr(data, 'node_types'):
            data.node_types = ['paper']
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
    elif dataset_name == 'pubmed':
        print("[REMOTE INFO] Loading PubMed...", flush=True)
        dataset = Planetoid(root="/tmp/PubMed", name="PubMed")
        data = dataset[0]
        if not hasattr(data, 'node_types'):
            data.node_types = ['paper']
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
    elif dataset_name == 'yelp':
        print("[REMOTE INFO] Loading Yelp...", flush=True)
        dataset = Yelp(root="/tmp/Yelp")
        data = dataset[0]
        if not hasattr(data, 'node_types'):
            data.node_types = ['review']
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
    elif dataset_name == 'physics':
        print("[REMOTE INFO] Loading Coauthor Physics...", flush=True)
        dataset = Coauthor(root="/tmp/CoauthorPhysics", name="Physics")
        data = dataset[0]
        if not hasattr(data, 'node_types'):
            data.node_types = ['author']
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
    elif dataset_name == 'citeseer':
        print("[REMOTE INFO] Loading CiteSeer...", flush=True)
        dataset = Planetoid(root="/tmp/CiteSeer", name="CiteSeer")
        data = dataset[0]
        if not hasattr(data, 'node_types'):
            data.node_types = ['paper']
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
        data.global_id = torch.arange(data.num_nodes)
    elif dataset_name == 'flickr':
        print("[REMOTE INFO] Loading Flickr...", flush=True)
        dataset = Flickr(root="/tmp/Flickr")
        data = dataset[0]
        if not hasattr(data, 'node_types'):
            data.node_types = ['image']
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
        data.global_id = torch.arange(data.num_nodes)
        
    print("\n[INFO] Symmetrizing full graph with SparseTensor...", flush=True)
    data.edge_index = make_undirected_fast(data.edge_index, data.num_nodes)
    # HARD SANITY CHECK: Ensure no node index exceeds num_nodes - 1 (causes C++ segfault in random_walk)
    max_node = data.edge_index.max().item()
    if max_node >= data.num_nodes:
        raise ValueError(f"CRITICAL: edge_index has out-of-bounds node {max_node}. Max allowed is {data.num_nodes - 1}.")
    print(f"  - Undirected edges: {data.edge_index.size(1)}", flush=True)

    # print(f"[INFO] Moving entire graph to GPU: {device}...", flush=True)
    # data = data.to(device)
    print(f"[INFO] Graph loaded. Keeping structure on CPU to optimize memory usage.", flush=True)
    # OPTIMIZATION: Keep data on CPU to save VRAM and utilize system RAM.
    # GPU is only used for model forward/backward and small batch data.

    base_feat_dim = data.x.size(1)
    print(f"\n[INFO] Base features: {base_feat_dim}", flush=True)
    feature_schema = getattr(data, "feature_schema", "homogeneous_raw")
    print(f"[INFO] Feature schema: {feature_schema}", flush=True)

    # --- OPTIMIZATION: USE SPARSETENSOR INSTEAD OF DICT ---
    print("[SETUP] Building SparseTensor adjacency for efficient slicing (on CPU)...", flush=True)
    # create sparse tensor and KEEP ON CPU
    adj_t = SparseTensor(
        row=data.edge_index[0], 
        col=data.edge_index[1], 
        sparse_sizes=(data.num_nodes, data.num_nodes)
    )
    # Pre-process CSR for fast lookup
    adj_t.csr() 
    print("  - SparseTensor built and on CPU.", flush=True)

    encoder = ImprovedSubgraphEncoder(base_feat_dim, 256, 128, dropout=0.1, use_residual=not disable_residual).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate) # Removed augmentor chaining
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=min_learning_rate,
            verbose=True,
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cosine_t_max if cosine_t_max > 0 else epochs),
            eta_min=min_learning_rate,
        )
    elif scheduler_type in {"none", "off", ""}:
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_type={scheduler_type!r}; use plateau, cosine, or none")
    encoder.train()
    
    # Hierarchy building 
    # Define Partition Configs (Gold Standard Targets)
    PARTITION_CONFIGS = {
        'cora': {'coarse': 20, 'fine': 5},
        'arxiv': {'coarse': 200, 'fine': 5},
        'mag': {'coarse': 2000, 'fine': 5},
        'pubmed': {'coarse': 20, 'fine': 5},
        'yelp': {'coarse': 700, 'fine': 5},
        'physics': {'coarse': 35, 'fine': 5},
        'citeseer': {'coarse': 10, 'fine': 5},
        'flickr': {'coarse': 100, 'fine': 5},
        'default': {'coarse': 100, 'fine': 10}
    }
    cfg = PARTITION_CONFIGS.get(dataset_name, PARTITION_CONFIGS['default'])
    target_coarse = cfg['coarse']; target_fine = cfg['fine']
    print(f"[CONFIG] Target Partitioning for {dataset_name}: Coarse={target_coarse}, Fine={target_fine}")

    hierarchy_cache_schema = (
        "type_rel_2000_fine5_finecov_v1" if dataset_name == "mag" else "finecov_v1"
    )
    CACHE_PATH = f"/cache/{dataset_name}_hierarchies_{hierarchy_cache_schema}.pt"
    if os.path.exists(CACHE_PATH):
        print(f"[CACHE] Found cached hierarchies at {CACHE_PATH}. Loading...", flush=True)
        try:
            hierarchies = torch.load(CACHE_PATH)
            if len(hierarchies) < num_hierarchies:
                print(f"[CACHE] Cache has {len(hierarchies)} hierarchies but {num_hierarchies} requested. Rebuilding...", flush=True)
                hierarchies = build_multiple_hierarchies_target(data, dataset_name, num_hierarchies)
                torch.save(hierarchies, CACHE_PATH)
                try:
                    volume = modal.Volume.from_name("jigsaw-cache-vol")
                    volume.commit()
                except: pass
            else:
                print(f"[CACHE] Successfully loaded {len(hierarchies)} hierarchies!", flush=True)
        except Exception as e:
            print(f"[CACHE] Failed to load cache: {e}. Re-building...", flush=True)
            hierarchies = build_multiple_hierarchies_target(data, dataset_name, num_hierarchies)
            print(f"[CACHE] Saving hierarchies to {CACHE_PATH}...", flush=True)
            torch.save(hierarchies, CACHE_PATH)
            try:
                # Force sync if using Modal Volume
                volume = modal.Volume.lookup("jigsaw-cache-vol")
                volume.commit() 
            except: pass
    else:
        print(f"[CACHE] No cache found at {CACHE_PATH}. Building from scratch...", flush=True)
        hierarchies = build_multiple_hierarchies_target(data, dataset_name, num_hierarchies)
        print(f"[CACHE] Saving hierarchies to {CACHE_PATH}...", flush=True)
        torch.save(hierarchies, CACHE_PATH)
        try:
             # Force sync if using Modal Volume
             volume = modal.Volume.lookup("jigsaw-cache-vol")
             volume.commit() 
        except: pass
    print("-" * 50)

    # NOW move to CPU for parallel worker processes to avoid VRAM hogging
    print("[INFO] Creating Graph and SparseTensor copies on CPU for parallel sampling...", flush=True)
    data_cpu = data.cpu()
    adj_t_cpu = adj_t.cpu()
    
    # Update hierarchy maps if they are tensors on GPU?
    # build_single_hierarchy leaves tensors on the device of `data` (GPU).
    # We need to move them to CPU too if we want workers to pickle them efficiently without CUDA context issues.
    # CRITICAL FIX: We must NOT modify 'hierarchies' in place because dataset_gpu needs them on GPU.
    # We create a deep copy for CPU
    import copy
    print("[INFO] Creating independent CPU hierarchy copy...", flush=True)
    hierarchies_cpu = []
    
    for h_gpu in hierarchies:
        h_cpu = {}
        for k, v in h_gpu.items():
            if isinstance(v, torch.Tensor):
                h_cpu[k] = v.cpu()
            elif isinstance(v, list):
                # Check list of Data objects
                if len(v) > 0:
                    # Find first valid Data object for type checking
                    first_valid = next((item for item in v if item is not None), None)
                    if first_valid is not None and isinstance(first_valid, Data):
                        # Data objects (coarse_graphs, fine_graphs)
                        # Explicitly handle attributes including adj_t
                        new_list = []
                        for item in v:
                            if item is None: 
                                new_list.append(None)
                                continue
                            item_cpu = item.cpu() # Moves standard attributes
                            # Manually move adj_t if present
                            if hasattr(item, 'adj_t') and item.adj_t is not None:
                                try:
                                    from torch_sparse import SparseTensor
                                    if hasattr(item_cpu, 'edge_index') and item_cpu.edge_index is not None:
                                        item_cpu.adj_t = SparseTensor(
                                            row=item_cpu.edge_index[0], 
                                            col=item_cpu.edge_index[1], 
                                            sparse_sizes=(item_cpu.num_nodes, item_cpu.num_nodes)
                                        )
                                        item_cpu.adj_t.csr()
                                    else:
                                        item_cpu.adj_t = item.adj_t.cpu()
                                except Exception:
                                    pass
                            new_list.append(item_cpu)
                        h_cpu[k] = new_list
                    elif hasattr(v[0], 'cpu'):
                        h_cpu[k] = [item.cpu() for item in v]
                    else:
                        h_cpu[k] = v # Copy list of fallback
                else: 
                     h_cpu[k] = []
            elif isinstance(v, nx.Graph):
                 # networkx Graph for coarse_part_graph
                 h_cpu[k] = copy.deepcopy(v)
            elif isinstance(v, dict):
                 # fine_graphs_dict or part_nodes_map
                 new_dict = {}
                 for subk, subv in v.items():
                     if isinstance(subv, torch.Tensor):
                         new_dict[subk] = subv.cpu()
                     elif isinstance(subv, Data):
                         # Handle Data objects in dictionary (v is fine_graphs_dict)
                         item_cpu = subv.cpu()
                         if hasattr(subv, 'adj_t') and subv.adj_t is not None:
                             try:
                                 from torch_sparse import SparseTensor
                                 if hasattr(item_cpu, 'edge_index') and item_cpu.edge_index is not None:
                                     item_cpu.adj_t = SparseTensor(
                                         row=item_cpu.edge_index[0], 
                                         col=item_cpu.edge_index[1], 
                                         sparse_sizes=(item_cpu.num_nodes, item_cpu.num_nodes)
                                     )
                                     item_cpu.adj_t.csr()
                                 else:
                                     item_cpu.adj_t = subv.adj_t.cpu()
                             except: pass
                         new_dict[subk] = item_cpu
                     else:
                         new_dict[subk] = subv
                 h_cpu[k] = new_dict
            else:
                 h_cpu[k] = v # ints, floats, etc.
        hierarchies_cpu.append(h_cpu)
    
    # FREE GPU HIERARCHY AND DATA IMMEDIATELY — they are large and no longer needed on GPU
    # coarse_part_nodes_map and adj_t are duplicated in hierarchies_cpu and adj_t_cpu
    print("[INFO] Releasing GPU hierarchy and data copies to save VRAM/RAM...", flush=True)
    del hierarchies
    del data
    gc.collect()
    torch.cuda.empty_cache()
    print("[INFO] Released GPU hierarchy and data copies.", flush=True)
    
    
    # 3. Create dataset for CPU workers
    # Use the CPU copies we just created
    coarse_graph_data_cache = {} if cache_partition_graphs else None
    sample_kwargs = {
        'prob_k_hop': prob_k_hop,
        'prob_single_part': prob_single_part,
        'prob_multi_coarse': prob_multi_coarse,
        'prob_random_walk': prob_random_walk,
        'prob_degree_k_hop': prob_degree_k_hop,
        'max_gpos_nodes': max_gpos_nodes,
        'max_train_coarse_parts': max_train_coarse_parts,
        'query_target_sizes': query_target_sizes,
        'query_size_jitter': query_size_jitter,
        'coarse_graph_data_cache': coarse_graph_data_cache,
        'hard_negative_source': hard_negative_source,
    }
    dataset_cpu = JigsawDataset(
        data_cpu, adj_t_cpu, hierarchies_cpu, batch_size,
        steps_per_epoch * batch_size, sample_kwargs=sample_kwargs
    )

    # --- CHECKPOINT RESUME LOGIC ---
    # User Request: Delete previous checkpoint for a fresh start if fresh_start=True
    if fresh_start and os.path.exists(checkpoint_path):
        print(f"[CLEANUP] Deleting previous checkpoint at {checkpoint_path} for a fresh start as requested.", flush=True)
        try:
            os.remove(checkpoint_path)
            # Commit to Modal Volume if needed (checkpoint_path is likely a volume mount)
            try:
                volume = modal.Volume.from_name("jigsaw-cache-vol")
                volume.commit()
            except: pass
        except Exception as e:
            print(f"[CLEANUP] Failed to delete checkpoint: {e}", flush=True)

    start_epoch = 0
    checkpoint_to_load = checkpoint_path if os.path.exists(checkpoint_path) else ""
    if not checkpoint_to_load and resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint_to_load = resume_from_checkpoint

    if checkpoint_to_load:
        print(f"[RESUME] Found checkpoint at {checkpoint_to_load}. Loading...", flush=True)
        try:
            checkpoint = torch.load(checkpoint_to_load)
            encoder_state = checkpoint.get("encoder_state_dict", checkpoint.get("encoder"))
            if encoder_state is None:
                raise KeyError("checkpoint has neither encoder_state_dict nor encoder")
            encoder.load_state_dict(encoder_state)
            checkpoint_supports_full_resume = (
                "optimizer_state_dict" in checkpoint and "epoch" in checkpoint
            )
            load_model_only = resume_model_only or not checkpoint_supports_full_resume
            if load_model_only:
                print(
                    "[RESUME] Loaded encoder weights only; optimizer/scheduler are reset "
                    "for this run's LR schedule.",
                    flush=True,
                )
                for pg in optimizer.param_groups:
                    pg['lr'] = learning_rate
                source_epoch = checkpoint.get('epoch', -1) + 1
                start_epoch = 0
                global_step = 0
                print(
                    f"[RESUME] Source checkpoint was at epoch {source_epoch}; "
                    "starting this model-only fine-tune from epoch 0.",
                    flush=True,
                )
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                # RESTORE global_step to prevent warmup reset
                global_step = checkpoint.get('global_step', start_epoch * steps_per_epoch)
            print(f"[RESUME] Resumed from Epoch {start_epoch}. global_step={global_step}", flush=True)
            if checkpoint_to_load != checkpoint_path:
                print(f"[RESUME] Source checkpoint differs; future saves go to {checkpoint_path}", flush=True)
        except Exception as e:
            print(f"[RESUME] Failed to load checkpoint: {e}. Starting from scratch.", flush=True)
    else:
        print("[RESUME] No checkpoint found. Starting from scratch.", flush=True)

    # --- DATALOADER (Move outside epoch loop for persistence) ---
    num_workers = 0 #else crash
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'collate_fn': jigsaw_collate_fn,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else None
    }
    batch_loader = DataLoader(dataset_cpu, **loader_kwargs)

    # --- PARTITION COVERAGE LOSS SETUP ---
    coarse_emb_cache = {} # coarse_id -> embedding tensor
    fine_emb_cache = {} # fine_id -> embedding tensor
    CACHE_REFRESH_STEPS = cache_refresh_steps
    FINE_CACHE_REFRESH_STEPS = max(0, fine_cache_refresh_steps)
    GAMMA_PARTITION = gamma_partition
    GAMMA_FINE_PARTITION = gamma_fine_partition
    # Use CPU version of hierarchies (since GPU version was deleted to save memory)
    coarse_part_nodes_map = hierarchies_cpu[0]['coarse_part_nodes_map']
    fine_part_nodes_map = hierarchies_cpu[0].get('fine_part_nodes_map', {})

    def refresh_embedding_cache(cache, part_nodes_map, label, clear_first=False, graph_data_cache=None):
        if clear_first:
            cache.clear()
        print(f"[CACHE] Refreshing {label} embedding cache for {len(part_nodes_map)} partitions...", flush=True)
        encoder.eval()

        pending_graphs = []
        pending_ids = []

        def flush_pending():
            if not pending_graphs:
                return
            try:
                batch = Batch.from_data_list(pending_graphs).to(device)
                embs, _ = encoder(batch.x, batch.edge_index, batch.batch)
                for part_id, emb in zip(pending_ids, embs):
                    cache[int(part_id)] = emb.detach().cpu()
                del batch, embs
            except Exception as batch_error:
                print(
                    f"[CACHE] Batched {label} encode failed; falling back to single graph encode: {batch_error}",
                    flush=True,
                )
                for part_id, graph in zip(pending_ids, pending_graphs):
                    try:
                        graph = graph.to(device)
                        graph_batch = torch.zeros(
                            graph.num_nodes, dtype=torch.long, device=device
                        )
                        emb, _ = encoder(graph.x, graph.edge_index, graph_batch)
                        cache[int(part_id)] = emb.squeeze(0).detach().cpu()
                        del graph, graph_batch, emb
                    except Exception:
                        continue
            finally:
                pending_graphs.clear()
                pending_ids.clear()

        with torch.no_grad():
            for i, (pid, nodes) in enumerate(part_nodes_map.items()):
                try:
                    pid_int = int(pid)
                    g = graph_data_cache.get(pid_int) if graph_data_cache is not None else None
                    if g is None:
                        g = _extract_subgraph_from_adj(adj_t_cpu, nodes, data_cpu)
                        if g is not None and graph_data_cache is not None:
                            graph_data_cache[pid_int] = g
                    if g is not None:
                        pending_graphs.append(g)
                        pending_ids.append(pid)
                        if len(pending_graphs) >= cache_encode_batch_size:
                            flush_pending()
                except Exception:
                    continue

                if (i + 1) % 50 == 0:
                    flush_pending()
                    gc.collect()
                    torch.cuda.empty_cache()
            flush_pending()
        encoder.train()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[CACHE] {label} embeddings available: {len(cache)}.", flush=True)

    def stack_embedding_cache(cache):
        ids_ordered = sorted(cache.keys())
        if cache:
            embs = torch.stack([cache[c] for c in ids_ordered])
            embs = embs.to(device).detach().clone().contiguous()
        else:
            embs = None
        id_to_idx = {c: i for i, c in enumerate(ids_ordered)} if cache else {}
        return ids_ordered, embs, id_to_idx

    def encode_live_positive_partitions(batch_metadata, id_to_index):
        """Re-encode a bounded set of true partitions with gradients enabled."""
        if max_live_positive_parts <= 0:
            return None, None
        counts = Counter(
            int(part_id)
            for metadata in batch_metadata
            for part_id in metadata.get(
                "coverage_coarse_ids", metadata.get("query_coarse_ids", [])
            )
            if int(part_id) in id_to_index
        )
        selected_ids = [
            part_id
            for part_id, _ in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[:max_live_positive_parts]
        ]
        graphs = []
        valid_ids = []
        for part_id in selected_ids:
            graph = None
            if coarse_graph_data_cache is not None:
                graph = coarse_graph_data_cache.get(int(part_id))
            if graph is None:
                graph = _extract_subgraph_from_adj(
                    adj_t_cpu, coarse_part_nodes_map[part_id], data_cpu
                )
                if graph is not None and coarse_graph_data_cache is not None:
                    coarse_graph_data_cache[int(part_id)] = graph
            if graph is not None:
                graphs.append(graph)
                valid_ids.append(part_id)
        if not graphs:
            return None, None
        live_batch = Batch.from_data_list(graphs).to(device)
        live_embeddings, _ = encoder(
            live_batch.x, live_batch.edge_index, live_batch.batch
        )
        live_indices = torch.tensor(
            [id_to_index[part_id] for part_id in valid_ids],
            dtype=torch.long,
            device=device,
        )
        return live_indices, live_embeddings

    # Initial cache population before training starts
    refresh_embedding_cache(
        coarse_emb_cache,
        coarse_part_nodes_map,
        "coarse",
        clear_first=True,
        graph_data_cache=coarse_graph_data_cache,
    )
    if GAMMA_FINE_PARTITION > 0.0:
        refresh_embedding_cache(fine_emb_cache, fine_part_nodes_map, "fine", clear_first=True)

    # Pre-build lookup structures for Partition Coverage Loss
    cids_ordered, coarse_part_embs, cid_to_idx = stack_embedding_cache(coarse_emb_cache)
    fids_ordered, fine_part_embs, fid_to_idx = stack_embedding_cache(fine_emb_cache)

    def build_fixed_validation_queries(count, seed):
        """Build deterministic 20-node k-hop queries aligned with paper evaluation."""
        if count <= 0:
            return []
        from torch_geometric.utils import k_hop_subgraph

        python_state = random.getstate()
        torch_state = torch.get_rng_state()
        random.seed(seed)
        torch.manual_seed(seed)
        node_to_coarse = hierarchies_cpu[0]["node_to_coarse_map"]
        row_ptr, columns, _ = adj_t_cpu.csr()
        validation = []
        attempts = 0
        while len(validation) < count and attempts < count * 100:
            attempts += 1
            anchor = random.randint(0, data_cpu.num_nodes - 1)
            try:
                allowed, _, _, _ = k_hop_subgraph(
                    anchor,
                    num_hops=3,
                    edge_index=data_cpu.edge_index,
                    relabel_nodes=False,
                    num_nodes=data_cpu.num_nodes,
                )
            except Exception:
                continue
            allowed_set = set(int(node) for node in allowed.tolist())
            queue_nodes = deque([anchor])
            visited = {anchor}
            selected = []
            while queue_nodes and len(selected) < 20:
                node = queue_nodes.popleft()
                selected.append(node)
                start = int(row_ptr[node].item())
                end = int(row_ptr[node + 1].item())
                neighbors = [
                    int(neighbor)
                    for neighbor in columns[start:end].tolist()
                    if int(neighbor) in allowed_set and int(neighbor) not in visited
                ]
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    visited.add(neighbor)
                    queue_nodes.append(neighbor)
            if len(selected) < 10:
                continue
            nodes = torch.tensor(selected, dtype=torch.long)
            graph = _extract_subgraph_from_adj(adj_t_cpu, nodes, data_cpu)
            true_ids = sorted(
                {
                    int(node_to_coarse[int(node)])
                    for node in nodes.tolist()
                    if int(node) in node_to_coarse
                }
            )
            if graph is not None and true_ids:
                validation.append((graph, true_ids))
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        if len(validation) != count:
            raise RuntimeError(
                f"Generated only {len(validation)}/{count} fixed validation queries"
            )
        print(
            f"[VALIDATION] Built {len(validation)} fixed k-hop queries with seed={seed}.",
            flush=True,
        )
        return validation

    def evaluate_fixed_validation(validation):
        if not validation or coarse_part_embs is None:
            return {}
        encoder.eval()
        with torch.no_grad():
            query_batch = Batch.from_data_list([item[0] for item in validation]).to(device)
            query_embeddings, _ = encoder(
                query_batch.x, query_batch.edge_index, query_batch.batch
            )
            scores = torch.matmul(query_embeddings, coarse_part_embs.T)
        metrics = {}
        for topk in validation_topks:
            k = min(topk, scores.shape[1])
            selected = torch.topk(scores, k, dim=1).indices.detach().cpu().tolist()
            fullcov = 0
            recalls = []
            for selected_ids, (_, true_actual_ids) in zip(selected, validation):
                true_ids = {
                    cid_to_idx[part_id]
                    for part_id in true_actual_ids
                    if part_id in cid_to_idx
                }
                selected_set = set(int(item) for item in selected_ids)
                covered = len(true_ids & selected_set)
                recalls.append(covered / len(true_ids) if true_ids else 0.0)
                fullcov += int(bool(true_ids) and true_ids.issubset(selected_set))
            metrics[f"fullcov_at_{topk}"] = fullcov
            metrics[f"recall_at_{topk}"] = sum(recalls) / len(recalls)
        encoder.train()
        return metrics

    fixed_validation = []
    for seed in validation_seeds:
        fixed_validation.extend(build_fixed_validation_queries(validation_queries, seed))
    best_validation_key = tuple([-1] * len(validation_topks) + [-1.0] * len(validation_topks))
    validations_without_improvement = 0
    should_stop_early = False
    suffix = "" if run_name in ("", "default") else f"_{run_name}"
    best_fullcov_model_path = (
        f"/cache/models/{dataset_name}-6_layer-model-jigsaw{suffix}_best_fullcov.pth"
    )
    baseline_validation_metrics = evaluate_fixed_validation(fixed_validation)
    if baseline_validation_metrics:
        best_validation_key = tuple(
            baseline_validation_metrics[f"fullcov_at_{topk}"]
            for topk in validation_topks
        ) + tuple(
            baseline_validation_metrics[f"recall_at_{topk}"]
            for topk in validation_topks
        )
        print(
            "[VALIDATION BASELINE] "
            + " ".join(
                f"FullCov@{topk}={baseline_validation_metrics[f'fullcov_at_{topk}']}/"
                f"{len(fixed_validation)} "
                f"Recall@{topk}={baseline_validation_metrics[f'recall_at_{topk}']:.4f}"
                for topk in validation_topks
            ),
            flush=True,
        )
        os.makedirs("/cache/models", exist_ok=True)
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "validation_metrics": baseline_validation_metrics,
                "validation_epoch": 0,
            },
            best_fullcov_model_path,
        )

    # --- LR WARMUP (Manual implementation to avoid scheduler conflict) ---
    WARMUP_STEPS = max(0, warmup_steps)
    if 'global_step' not in locals():
        global_step = start_epoch * steps_per_epoch
    BASE_LR = learning_rate
    last_validation_metrics = baseline_validation_metrics

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        total_part_loss = 0
        total_fine_part_loss = 0
        iterator = iter(batch_loader)
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}", unit="step", mininterval=30.0) 
        
        for step in pbar:
            global_step += 1
            
            # --- CACHE REFRESH (Every 50 steps) ---
            if global_step % CACHE_REFRESH_STEPS == 0:
                refresh_embedding_cache(
                    coarse_emb_cache,
                    coarse_part_nodes_map,
                    "coarse",
                    graph_data_cache=coarse_graph_data_cache,
                )
                cids_ordered, coarse_part_embs, cid_to_idx = stack_embedding_cache(coarse_emb_cache)

            if (
                GAMMA_FINE_PARTITION > 0.0
                and FINE_CACHE_REFRESH_STEPS > 0
                and global_step % FINE_CACHE_REFRESH_STEPS == 0
            ):
                refresh_embedding_cache(fine_emb_cache, fine_part_nodes_map, "fine")
                fids_ordered, fine_part_embs, fid_to_idx = stack_embedding_cache(fine_emb_cache)
            # Linear warmup
            if WARMUP_STEPS > 0 and global_step <= WARMUP_STEPS:
                warmup_factor = global_step / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg['lr'] = BASE_LR * warmup_factor
            
            
            try:
                res = next(iterator)
            except StopIteration:
                iterator = iter(batch_loader)
                res = next(iterator)

            # Defensive check for invalid batch
            if not res or not isinstance(res, tuple) or len(res) < 5 or res[0] is None:
                continue

            batch_data = res[0], res[1], res[2]
            batch_metadata = res[3]
            hns_list = res[4]
            
            # Log summary less frequently (every 50 steps)
            if step % 50 == 0:
                avg_gen_time = sum(m.get('time', 0) for m in batch_metadata) / len(batch_metadata)
                type_counts = Counter(m.get('type', 'unknown') for m in batch_metadata)
                counts_str = ", ".join([f"{k}:{v}" for k, v in type_counts.items()])
                coverage_counts = [
                    len(m.get('coverage_coarse_ids', m.get('query_coarse_ids', [])))
                    for m in batch_metadata
                ]
                fine_coverage_counts = [
                    len(m.get('coverage_fine_ids', m.get('query_fine_ids', [])))
                    for m in batch_metadata
                ]
                context_counts = [len(m.get('query_coarse_ids', [])) for m in batch_metadata]
                query_node_counts = [
                    int(m.get('query_node_count', 0)) for m in batch_metadata
                ]
                impossible_at_20 = sum(1 for count in coverage_counts if count > 20)
                impossible_at_50 = sum(1 for count in coverage_counts if count > 50)
                tqdm.write(
                    f"[Step {step}] GenAvg:{avg_gen_time:.3f}s | Types: {{ {counts_str} }} "
                    f"| QueryNodes avg/max={sum(query_node_counts)/len(query_node_counts):.1f}/{max(query_node_counts)} "
                    f"| CoverageTargets avg/max={sum(coverage_counts)/len(coverage_counts):.1f}/{max(coverage_counts)} "
                    f"| FineTargets avg/max={sum(fine_coverage_counts)/len(fine_coverage_counts):.1f}/{max(fine_coverage_counts)} "
                    f"| ContextTargets avg/max={sum(context_counts)/len(context_counts):.1f}/{max(context_counts)} "
                    f"| Impossible@20/50={impossible_at_20}/{impossible_at_50}"
                )
            
            # Move batches to GPU with robust OOM handling
            try:
                query_batch = batch_data[0].to(device)
                pos_batch = batch_data[1].to(device)
                coarse_pos_batch = batch_data[2].to(device)
                
                optimizer.zero_grad()
                
                zq, zq_nodes = encoder(query_batch.x, query_batch.edge_index, query_batch.batch)
                z_pos, z_pos_nodes = encoder(pos_batch.x, pos_batch.edge_index, pos_batch.batch)
                z_coarse, _ = encoder(coarse_pos_batch.x, coarse_pos_batch.edge_index, coarse_pos_batch.batch)
                
                z_hn_matrix = None
                if hard_negative_source == "none":
                    z_hn_matrix = None
                elif hard_negative_source == "cache" and coarse_part_embs is not None:
                    hn_id_lists = [
                        [
                            cid_to_idx[int(hn_id)]
                            for hn_id in metadata.get("hard_negative_coarse_ids", [])
                            if int(hn_id) in cid_to_idx
                        ]
                        for metadata in batch_metadata
                    ]
                    max_hns = max((len(ids) for ids in hn_id_lists), default=0)
                    if max_hns > 0:
                        D = coarse_part_embs.shape[1]
                        z_hn_matrix = torch.zeros(
                            (len(hn_id_lists), max_hns, D),
                            device=device,
                            dtype=coarse_part_embs.dtype,
                        )
                        for row_idx, hn_indices in enumerate(hn_id_lists):
                            if hn_indices:
                                index_tensor = torch.tensor(
                                    hn_indices, dtype=torch.long, device=device
                                )
                                z_hn_matrix[row_idx, : len(hn_indices), :] = coarse_part_embs[index_tensor]
                else:
                    flat_hns = []
                    hn_counts = []
                    for hns in hns_list:
                        if hns:
                            hn_counts.append(len(hns))
                            flat_hns.extend(hns)
                        else:
                            hn_counts.append(0)
                    hn_batch = Batch.from_data_list(flat_hns).to(device) if flat_hns else None
                    if hn_batch is not None:
                        z_hns_flat, _ = encoder(hn_batch.x, hn_batch.edge_index, hn_batch.batch)
                        B = len(hn_counts)
                        D = z_hns_flat.shape[1]
                        max_hns = max(hn_counts)
                        z_hn_matrix = torch.zeros((B, max_hns, D), device=device)
                        idx = 0
                        for i, count in enumerate(hn_counts):
                            if count > 0:
                                z_hn_matrix[i, :count, :] = z_hns_flat[idx:idx+count]
                                idx += count
                
                loss = hierarchical_info_nce_loss(
                    zq, z_pos, z_coarse, zq_nodes, z_pos_nodes,
                    query_batch, pos_batch, hard_negatives=z_hn_matrix,
                    alpha=alpha, beta=beta
                )

                # --- PARTITION COVERAGE LOSS ---
                loss_partition = torch.tensor(0.0, device=device)
                loss_fine_partition = torch.tensor(0.0, device=device)
                live_partition_indices = None
                live_partition_embeddings = None
                if coarse_part_embs is not None and len(coarse_part_embs) > 1:
                    query_coarse_ids_remapped = [
                        [
                            cid_to_idx[c]
                            for c in m.get('coverage_coarse_ids', m.get('query_coarse_ids', []))
                            if c in cid_to_idx
                        ]
                        for m in batch_metadata
                    ]
                    (
                        live_partition_indices,
                        live_partition_embeddings,
                    ) = encode_live_positive_partitions(batch_metadata, cid_to_idx)
                    loss_partition = partition_coverage_loss(
                        zq, coarse_part_embs, query_coarse_ids_remapped,
                        temperature=coverage_temperature,
                        target_topk=coverage_topk,
                        topk_bucket_size=coverage_topk_bucket_size,
                        topk_weight=coverage_topk_weight,
                        topk_margin=coverage_topk_margin,
                        positive_aggregation=coverage_positive_aggregation,
                        cvar_fraction=coverage_cvar_fraction,
                        smoothmax_temperature=coverage_smoothmax_temperature,
                        live_partition_indices=live_partition_indices,
                        live_partition_embeddings=live_partition_embeddings,
                    )
                    loss = loss + GAMMA_PARTITION * loss_partition

                if (
                    GAMMA_FINE_PARTITION > 0.0
                    and fine_part_embs is not None
                    and len(fine_part_embs) > 1
                ):
                    query_fine_ids_remapped = [
                        [
                            fid_to_idx[f]
                            for f in m.get('coverage_fine_ids', m.get('query_fine_ids', []))
                            if f in fid_to_idx
                        ]
                        for m in batch_metadata
                    ]

                    loss_fine_partition = partition_coverage_loss(
                        zq, fine_part_embs, query_fine_ids_remapped,
                        temperature=coverage_temperature,
                        target_topk=coverage_topk,
                        topk_bucket_size=coverage_topk_bucket_size,
                        topk_weight=coverage_topk_weight,
                        topk_margin=coverage_topk_margin,
                        positive_aggregation=coverage_positive_aggregation,
                        cvar_fraction=coverage_cvar_fraction,
                        smoothmax_temperature=coverage_smoothmax_temperature,
                    )
                    loss = loss + GAMMA_FINE_PARTITION * loss_fine_partition
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_part_loss += loss_partition.item()
                total_fine_part_loss += loss_fine_partition.item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "part": f"{loss_partition.item():.4f}",
                    "fine": f"{loss_fine_partition.item():.4f}",
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    tqdm.write(f"\n[OOM] Step {step}. Skipping batch.")
                    torch.cuda.empty_cache()
                else: raise e
            
            # Robust local cleanup
            try: del query_batch, pos_batch, coarse_pos_batch
            except NameError: pass
            try: del zq, z_pos, z_coarse, zq_nodes, z_pos_nodes
            except NameError: pass
            try: del z_hn_matrix, loss
            except NameError: pass
            try: del live_partition_indices, live_partition_embeddings
            except NameError: pass
            
            if step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
        avg_loss = total_loss / steps_per_epoch if steps_per_epoch > 0 else 0
        avg_part_loss = total_part_loss / steps_per_epoch if steps_per_epoch > 0 else 0
        avg_fine_part_loss = total_fine_part_loss / steps_per_epoch if steps_per_epoch > 0 else 0
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        
        gc.collect()
        torch.cuda.empty_cache()
        current_lr = optimizer.param_groups[0]["lr"]
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_res = torch.cuda.memory_reserved() / 1e9
        print(
            f"Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.6f}, "
            f"CoarsePart = {avg_part_loss:.6f}, FinePart = {avg_fine_part_loss:.6f}, "
            f"LR = {current_lr:.1e}, GPU Mem: {mem_alloc:.2f}/{mem_res:.2f} GB"
        )

        if fixed_validation and (
            (epoch + 1) % validation_interval == 0 or (epoch + 1) == epochs
        ):
            refresh_embedding_cache(
                coarse_emb_cache,
                coarse_part_nodes_map,
                "coarse-validation",
                graph_data_cache=coarse_graph_data_cache,
            )
            cids_ordered, coarse_part_embs, cid_to_idx = stack_embedding_cache(
                coarse_emb_cache
            )
            last_validation_metrics = evaluate_fixed_validation(fixed_validation)
            print(
                "[VALIDATION] "
                + " ".join(
                    f"FullCov@{topk}={last_validation_metrics[f'fullcov_at_{topk}']}/"
                    f"{len(fixed_validation)} "
                    f"Recall@{topk}={last_validation_metrics[f'recall_at_{topk}']:.4f}"
                    for topk in validation_topks
                ),
                flush=True,
            )
            validation_key = tuple(
                last_validation_metrics[f"fullcov_at_{topk}"]
                for topk in validation_topks
            ) + tuple(
                last_validation_metrics[f"recall_at_{topk}"]
                for topk in validation_topks
            )
            if validation_key > best_validation_key:
                best_validation_key = validation_key
                validations_without_improvement = 0
                os.makedirs("/cache/models", exist_ok=True)
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "validation_metrics": last_validation_metrics,
                        "validation_epoch": epoch + 1,
                    },
                    best_fullcov_model_path,
                )
                print(
                    f"[VALIDATION] New best FullCov model saved to {best_fullcov_model_path}",
                    flush=True,
                )
                try:
                    modal.Volume.from_name("jigsaw-cache-vol").commit()
                except Exception as e:
                    print(f"[VALIDATION] Volume commit failed (non-fatal): {e}", flush=True)
            else:
                validations_without_improvement += 1
                print(
                    f"[VALIDATION] No improvement for "
                    f"{validations_without_improvement}/{early_stopping_patience or 'off'} checks.",
                    flush=True,
                )
                should_stop_early = (
                    early_stopping_patience > 0
                    and validations_without_improvement >= early_stopping_patience
                )

        # --- CHECKPOINT SAVE LOGIC ---
        if (epoch + 1) % checkpoint_interval_epochs == 0 or (epoch + 1) == epochs:
            print(f"[CHECKPOINT] Saving checkpoint to {checkpoint_path}...", flush=True)
            save_dict = {
                'epoch': epoch,
                'global_step': global_step,
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'training_config': {
                    'dataset': dataset_name,
                    'epochs': epochs,
                    'steps_per_epoch': steps_per_epoch,
                    'batch_size': batch_size,
                    'num_hierarchies': num_hierarchies,
                    'gamma_partition': gamma_partition,
                    'gamma_fine_partition': gamma_fine_partition,
                    'coverage_temperature': coverage_temperature,
                    'coverage_topk': coverage_topk,
                    'coverage_topk_bucket_size': coverage_topk_bucket_size,
                    'coverage_topk_weight': coverage_topk_weight,
                    'coverage_topk_margin': coverage_topk_margin,
                    'coverage_positive_aggregation': coverage_positive_aggregation,
                    'coverage_cvar_fraction': coverage_cvar_fraction,
                    'coverage_smoothmax_temperature': coverage_smoothmax_temperature,
                    'max_live_positive_parts': max_live_positive_parts,
                    'fine_cache_refresh_steps': fine_cache_refresh_steps,
                    'alpha': alpha,
                    'beta': beta,
                    'prob_k_hop': prob_k_hop,
                    'prob_single_part': prob_single_part,
                    'prob_multi_coarse': prob_multi_coarse,
                    'prob_random_walk': prob_random_walk,
                    'prob_degree_k_hop': prob_degree_k_hop,
                    'hard_negative_source': hard_negative_source,
                    'max_gpos_nodes': max_gpos_nodes,
                    'max_train_coarse_parts': max_train_coarse_parts,
                    'query_target_sizes': query_target_sizes,
                    'query_size_jitter': query_size_jitter,
                    'cache_refresh_steps': cache_refresh_steps,
                    'cache_encode_batch_size': cache_encode_batch_size,
                    'cache_partition_graphs': cache_partition_graphs,
                    'checkpoint_interval_epochs': checkpoint_interval_epochs,
                    'feature_schema': feature_schema,
                    'learning_rate': learning_rate,
                    'scheduler_type': scheduler_type,
                    'min_learning_rate': min_learning_rate,
                    'warmup_steps': warmup_steps,
                    'plateau_patience': plateau_patience,
                    'plateau_factor': plateau_factor,
                    'cosine_t_max': cosine_t_max,
                    'resume_model_only': resume_model_only,
                    'validation_queries': validation_queries,
                    'validation_interval': validation_interval,
                    'validation_seed': validation_seed,
                    'validation_seeds': validation_seeds,
                    'validation_topks': validation_topks,
                    'early_stopping_patience': early_stopping_patience,
                    'training_seed': training_seed,
                    'disable_residual': disable_residual,
                },
                'validation_metrics': last_validation_metrics,
                'best_validation_key': best_validation_key,
            }
            torch.save(save_dict, checkpoint_path)
            try:
                # Force sync to Modal Volume
                volume = modal.Volume.from_name("jigsaw-cache-vol")
                volume.commit()
                print("[CHECKPOINT] Volume committed successfully.", flush=True)
            except Exception as e:
                print(f"[CHECKPOINT] Volume commit failed (non-fatal): {e}", flush=True)
        if should_stop_early:
            print(
                f"[EARLY STOP] Stopping after epoch {epoch + 1}; "
                f"best validation key={best_validation_key}.",
                flush=True,
            )
            break

    print("\n[REMOTE INFO] Training finished.")
    suffix = "" if run_name in ("", "default") else f"_{run_name}"
    final_model_path = f"/cache/models/{dataset_name}-6_layer-model-jigsaw{suffix}.pth"
    training_config = {
        'dataset': dataset_name,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'batch_size': batch_size,
        'num_hierarchies': num_hierarchies,
        'gamma_partition': gamma_partition,
        'gamma_fine_partition': gamma_fine_partition,
        'coverage_temperature': coverage_temperature,
        'coverage_topk': coverage_topk,
        'coverage_topk_bucket_size': coverage_topk_bucket_size,
        'coverage_topk_weight': coverage_topk_weight,
        'coverage_topk_margin': coverage_topk_margin,
        'coverage_positive_aggregation': coverage_positive_aggregation,
        'coverage_cvar_fraction': coverage_cvar_fraction,
        'coverage_smoothmax_temperature': coverage_smoothmax_temperature,
        'max_live_positive_parts': max_live_positive_parts,
        'fine_cache_refresh_steps': fine_cache_refresh_steps,
        'alpha': alpha,
        'beta': beta,
        'prob_k_hop': prob_k_hop,
        'prob_single_part': prob_single_part,
        'prob_multi_coarse': prob_multi_coarse,
        'prob_random_walk': prob_random_walk,
        'prob_degree_k_hop': prob_degree_k_hop,
        'hard_negative_source': hard_negative_source,
        'max_gpos_nodes': max_gpos_nodes,
        'max_train_coarse_parts': max_train_coarse_parts,
        'query_target_sizes': query_target_sizes,
        'query_size_jitter': query_size_jitter,
        'cache_refresh_steps': cache_refresh_steps,
        'cache_encode_batch_size': cache_encode_batch_size,
        'cache_partition_graphs': cache_partition_graphs,
        'checkpoint_interval_epochs': checkpoint_interval_epochs,
        'feature_schema': feature_schema,
        'learning_rate': learning_rate,
        'scheduler_type': scheduler_type,
        'min_learning_rate': min_learning_rate,
        'warmup_steps': warmup_steps,
        'plateau_patience': plateau_patience,
        'plateau_factor': plateau_factor,
        'cosine_t_max': cosine_t_max,
        'resume_model_only': resume_model_only,
        'validation_queries': validation_queries,
        'validation_interval': validation_interval,
        'validation_seed': validation_seed,
        'validation_seeds': validation_seeds,
        'validation_topks': validation_topks,
        'early_stopping_patience': early_stopping_patience,
        'training_seed': training_seed,
        'validation_metrics': last_validation_metrics,
        'disable_residual': disable_residual,
    }
    try:
        os.makedirs("/cache/models", exist_ok=True)
        torch.save(
            {'encoder': encoder.cpu().state_dict(), 'training_config': training_config},
            final_model_path,
        )
        print(f"[MODEL] Final model saved to {final_model_path}", flush=True)
    except Exception as e:
        print(f"[MODEL] Final model save failed: {e}", flush=True)

    try:
        if log_file is not None:
            log_file.flush()
        volume = modal.Volume.from_name("jigsaw-cache-vol")
        volume.commit()
        print("[LOG] Training log committed to jigsaw-cache-vol.", flush=True)
    except Exception as e:
        print(f"[LOG] Final log commit failed (non-fatal): {e}", flush=True)
    return {'model_path': final_model_path, 'training_config': training_config}

# --- THE LOCAL ENTRYPOINT ---
@app.local_entrypoint()
def main(dataset: str = "mag", epochs: int = 100, batch_size: int = 32,
         steps_per_epoch: int = 50, num_hierarchies: int = 1,
         run_name: str = "default", fresh: bool = False,
         gamma_partition: float = 0.5, coverage_temperature: float = 0.05,
         coverage_topk: int = 0, coverage_topk_bucket_size: int = 10,
         coverage_topk_weight: float = 0.0,
         coverage_topk_margin: float = 0.0,
         coverage_positive_aggregation: str = "mean",
         coverage_cvar_fraction: float = 0.25,
         coverage_smoothmax_temperature: float = 0.1,
         max_live_positive_parts: int = 0,
         gamma_fine_partition: float = 0.0, fine_cache_refresh_steps: int = 250,
         alpha: float = 0.2, beta: float = 0.0,
         prob_k_hop: float = 0.35, prob_single_part: float = 0.15,
         prob_multi_coarse: float = 0.30,
         prob_random_walk: float = 0.0, prob_degree_k_hop: float = 0.0,
         hard_negative_source: str = "graphs",
         max_gpos_nodes: int = 4000,
         max_train_coarse_parts: int = 20, cache_refresh_steps: int = 20,
         query_target_sizes: str = "20,20,20,50,100",
         query_size_jitter: int = 5,
         cache_encode_batch_size: int = 1, cache_partition_graphs: int = 0,
         checkpoint_interval_epochs: int = 2,
         resume_from_checkpoint: str = "", learning_rate: float = 1e-4,
         scheduler_type: str = "plateau", min_learning_rate: float = 1e-5,
         warmup_steps: int = 100, plateau_patience: int = 10,
         plateau_factor: float = 0.5, cosine_t_max: int = 0,
         resume_model_only: bool = False, validation_queries: int = 0,
         validation_interval: int = 2, validation_seed: int = 31415,
         validation_seeds: str = "", validation_topks: str = "20,50,100",
         early_stopping_patience: int = 0,
         training_seed: int = 42,
         spawn: bool = False,
         disable_residual: bool = False):
    print(f"🚀 Starting Jigsaw GNN training on Modal for {dataset} (run={run_name}, Fresh Start: {fresh})...")
    suffix = "" if run_name in ("", "default") else f"_{run_name}"
    checkpoint_path = f"/cache/{dataset}{suffix}_checkpoint.pth"
    
    train_kwargs = dict(
        dataset_name=dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        num_hierarchies=num_hierarchies,
        checkpoint_path=checkpoint_path,
        fresh_start=fresh,
        run_name=run_name,
        gamma_partition=gamma_partition,
        coverage_temperature=coverage_temperature,
        coverage_topk=coverage_topk,
        coverage_topk_bucket_size=coverage_topk_bucket_size,
        coverage_topk_weight=coverage_topk_weight,
        coverage_topk_margin=coverage_topk_margin,
        coverage_positive_aggregation=coverage_positive_aggregation,
        coverage_cvar_fraction=coverage_cvar_fraction,
        coverage_smoothmax_temperature=coverage_smoothmax_temperature,
        max_live_positive_parts=max_live_positive_parts,
        gamma_fine_partition=gamma_fine_partition,
        fine_cache_refresh_steps=fine_cache_refresh_steps,
        alpha=alpha,
        beta=beta,
        prob_k_hop=prob_k_hop,
        prob_single_part=prob_single_part,
        prob_multi_coarse=prob_multi_coarse,
        prob_random_walk=prob_random_walk,
        prob_degree_k_hop=prob_degree_k_hop,
        hard_negative_source=hard_negative_source,
        max_gpos_nodes=max_gpos_nodes,
        max_train_coarse_parts=max_train_coarse_parts,
        query_target_sizes=query_target_sizes,
        query_size_jitter=query_size_jitter,
        cache_refresh_steps=cache_refresh_steps,
        cache_encode_batch_size=cache_encode_batch_size,
        cache_partition_graphs=cache_partition_graphs,
        checkpoint_interval_epochs=checkpoint_interval_epochs,
        resume_from_checkpoint=resume_from_checkpoint,
        learning_rate=learning_rate,
        scheduler_type=scheduler_type,
        min_learning_rate=min_learning_rate,
        warmup_steps=warmup_steps,
        plateau_patience=plateau_patience,
        plateau_factor=plateau_factor,
        cosine_t_max=cosine_t_max,
        resume_model_only=resume_model_only,
        validation_queries=validation_queries,
        validation_interval=validation_interval,
        validation_seed=validation_seed,
        validation_seeds=validation_seeds,
        validation_topks=validation_topks,
        early_stopping_patience=early_stopping_patience,
        training_seed=training_seed,
        disable_residual=disable_residual,
    )

    if spawn:
        call = train.spawn(**train_kwargs)
        print(f"✅ Spawned remote training call: {call.object_id}")
        print(f"   Checkpoint path: {checkpoint_path}")
        return

    result = train.remote(**train_kwargs)
    print(f"✅ Remote model saved to '{result.get('model_path')}'")
