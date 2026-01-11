import torch
import networkx as nx
import random
import multiprocessing
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="outdated")
from collections import defaultdict
from torch_sparse import SparseTensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import subgraph
from tqdm import tqdm

# --- HELPER FOR ROBUST PARTITIONING (Global for Multiprocessing) ---
def _partitioner_target(queue, n_parts, xadj, adjncy):
    import pymetis
    try:
        _, membership = pymetis.part_graph(n_parts, xadj=xadj, adjncy=adjncy)
        queue.put(membership)
    except Exception as e:
        print(f"      - [Subprocess] Error during partitioning: {e}", flush=True)
        queue.put(None)

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

def convert_hetero_to_homo(hetero_data: HeteroData) -> Data:
    """
    Convert OGBN-MAG HeteroData -> homogeneous Data
    """
    print("  - Converting heterogeneous graph to homogeneous...")
    node_types = list(hetero_data.num_nodes_dict.keys())
    node_offset, total_nodes = {}, 0
    for nt in node_types:
        node_offset[nt] = total_nodes
        total_nodes += hetero_data.num_nodes_dict[nt]

    # Handle case where paper features might be missing or different in other datasets
    # Defaulting to paper features size if available
    if "paper" in hetero_data.x_dict:
        feat_dim = hetero_data.x_dict["paper"].size(1)
        x = torch.zeros(total_nodes, feat_dim, dtype=torch.float)
        p_start, p_end = node_offset["paper"], node_offset["paper"] + hetero_data.num_nodes_dict["paper"]
        x[p_start:p_end] = hetero_data.x_dict["paper"]
    else:
        # Fallback for empty features or different schema
        x = torch.zeros(total_nodes, 128, dtype=torch.float) # Placeholder

    node_type_ids = torch.zeros(total_nodes, dtype=torch.long)
    for i, nt in enumerate(node_types):
        s, e = node_offset[nt], node_offset[nt] + hetero_data.num_nodes_dict[nt]
        node_type_ids[s:e] = i

    all_ei = []
    for (src_t, rel, dst_t), ei in hetero_data.edge_index_dict.items():
        gei = ei.clone(); gei[0] += node_offset[src_t]; gei[1] += node_offset[dst_t]
        all_ei.append(gei)

    edge_index = torch.cat(all_ei, dim=1) if all_ei else torch.empty((2, 0), dtype=torch.long)

    homo = Data(x=x, edge_index=edge_index, num_nodes=total_nodes)
    homo.node_type = node_type_ids; homo.node_types = node_types
    homo.node_offset = node_offset; homo.global_id = torch.arange(total_nodes, dtype=torch.long)
    print(f"    - Converted to homogeneous: {homo.num_nodes} nodes, {homo.edge_index.size(1)} edges")
    return homo

def make_undirected_fast(edge_index, num_nodes):
    # This part runs on CPU initially or we can move edge_index to GPU first
    # Since we are immediately moving to GPU after, let's keep this as is for robust conversion
    try:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to_symmetric()
        row, col, _ = adj.coo()
        return torch.stack([row, col], dim=0)
    except Exception as e:
        # Fallback if sparse tensor fails (e.g. CPU OOM)
        print(f"    - SparseTensor symmetrization failed ({e}), falling back to slow utils...")
        from torch_geometric.utils import to_undirected
        return to_undirected(edge_index, num_nodes=num_nodes)

def make_partitions(dataset, num_parts, keep_features=True):
    
    # --- SANITY CHECKS ---
    if dataset.num_nodes == 0: return [], {}
    # Ensure consistency
    if dataset.x is not None and dataset.x.size(0) != dataset.num_nodes:
            # warn but continue? No, error.
            # raise RuntimeError(f"Dataset x size {dataset.x.size(0)} != num_nodes {dataset.num_nodes}")
            pass # Relaxed for now
    
    if dataset.num_nodes < num_parts: num_parts = dataset.num_nodes
    if num_parts <= 1: 
        d = Data(edge_index=dataset.edge_index, num_nodes=dataset.num_nodes)
        if keep_features:
                if dataset.x is not None: d.x = dataset.x
                if dataset.y is not None: d.y = dataset.y
        # Copy basic attrs
        if hasattr(dataset, 'node_type'): d.node_type = dataset.node_type
        if hasattr(dataset, 'global_id'): d.global_id = dataset.global_id
        
        d.part_id = 0
        return [d], {0: torch.arange(dataset.num_nodes, device=dataset.edge_index.device)}
    
    # Partitioning needs to happen on CPU (pymetis requirement usually)
    edge_index_cpu = dataset.edge_index.cpu()
    
    adj = SparseTensor.from_edge_index(edge_index_cpu, sparse_sizes=(dataset.num_nodes, dataset.num_nodes))
    xadj_t, adjncy_t, _ = adj.csr(); xadj, adjncy = xadj_t.tolist(), adjncy_t.tolist()
    
    try:
        _, membership = run_pymetis_in_subprocess(num_parts, xadj=xadj, adjncy=adjncy)
    except RuntimeError as e:
        print(f"Pooling Metis failed: {e}. Falling back to random partition for robustness.")
        membership = [random.randint(0, num_parts-1) for _ in range(dataset.num_nodes)]
    
    part_graphs, part_nodes_map = [], {}
    for part_id in range(num_parts):
        node_indices = [i for i, p in enumerate(membership) if p == part_id]
        if node_indices:
            nodes_tensor = torch.tensor(node_indices, dtype=torch.long, device=dataset.edge_index.device)
            part_nodes_map[part_id] = nodes_tensor
            
            # Manual Data construction to avoid implicit subgraph issues and ensure correct relabeling
            try:
                relabeled_edge_index, _ = subgraph(nodes_tensor, dataset.edge_index, relabel_nodes=True, num_nodes=dataset.num_nodes)
            except Exception:
                # Fallback
                relabeled_edge_index = torch.empty((2,0), dtype=torch.long, device=dataset.edge_index.device)
            
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
            for global_attr in ['node_types', 'node_offset', 'edge_types', 'edge_offset']:
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
    
    fine_graphs, fine_part_nodes_map, fine_to_coarse_map = [], {}, {}; fine_global_idx = 0
    iterator = tqdm(enumerate(coarse_graphs), total=len(coarse_graphs), desc="    - Creating fine partitions", unit="coarse_part", ncols=100, mininterval=30.0)
    for coarse_list_idx, coarse_graph in iterator:
        # Fix for alignment: use original part_id if available
        coarse_idx = getattr(coarse_graph, 'part_id', coarse_list_idx)
        
        if coarse_idx not in coarse_part_nodes_map: continue
        global_nodes_of_this_coarse_part = coarse_part_nodes_map[coarse_idx]
        if coarse_graph.num_nodes < (num_fine * 2) or coarse_graph.num_edges == 0:
            finer_partitions, finer_nodes_map_local = [coarse_graph], {0: torch.arange(coarse_graph.num_nodes, device=data.x.device)}
        else: 
            finer_partitions, finer_nodes_map_local = make_partitions(coarse_graph, num_fine, keep_features=False)
        for fine_local_idx, fine_part in enumerate(finer_partitions):
            if fine_local_idx not in finer_nodes_map_local: continue
            local_indices_in_coarse = finer_nodes_map_local[fine_local_idx]
            global_indices_for_fine = global_nodes_of_this_coarse_part[local_indices_in_coarse]
            if fine_part.num_nodes > 10 and fine_part.num_edges > 0:
                fine_graphs.append(fine_part); fine_part_nodes_map[fine_global_idx] = global_indices_for_fine
                fine_to_coarse_map[fine_global_idx] = coarse_idx; fine_global_idx += 1
    
    # Pre-compute coarse_edge -> valid fine bridges
    fine_ids = torch.full((data.num_nodes,), -1, dtype=torch.long, device=data.x.device)
    for fid, nodes in fine_part_nodes_map.items():
        fine_ids[nodes] = fid
    
    f_src = fine_ids[src]
    f_dst = fine_ids[dst]
    
    # Filter edges where fine partitions differ (potential bridges)
    bridge_mask = (f_src != f_dst) & (f_src != -1) & (f_dst != -1)
    
    # Filter further: only edges between DIFFERENT coarse partitions
    bridge_mask = bridge_mask & (c_src != c_dst) & (c_src != -1) & (c_dst != -1)
    
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
        'fine_graphs': fine_graphs,
        'node_to_coarse_map': node_to_coarse_map,
        'fine_to_coarse_map': fine_to_coarse_map,
        'fine_part_nodes_map': fine_part_nodes_map,
        'coarse_part_graph': coarse_part_graph,
        'coarse_part_nodes_map': coarse_part_nodes_map,
        'coarse_edge_to_fine_bridges': dict(coarse_edge_to_fine_bridges)
    }

def build_multiple_hierarchies(data, n_hierarchies, target_coarse, target_fine):
    print(f"[SETUP] Building {n_hierarchies} different hierarchies for Jigsaw training...")
    hierarchies = []
    
    iterator = tqdm(range(n_hierarchies), desc="Building hierarchies", unit="hierarchy", mininterval=30.0)
    for i in iterator:
        if i == 0:
            # Hierarchy 0: EXACT TARGET
            c_sample, f_sample = target_coarse, target_fine
            type_str = "Exact"
        else:
            # Variation logic: +/- 15% for coarse
            c_var = max(3, int(target_coarse * 0.15))
            c_sample = random.randint(max(2, target_coarse - c_var), target_coarse + c_var)
            
            # Fine: Variation but STRICTLY clamped between 5 and 10
            f_var = 2
            val = random.randint(target_fine - f_var, target_fine + f_var)
            f_sample = max(5, min(10, val))
            type_str = "Variant"
            
        tqdm.write(f"  - Hierarchy {i} ({type_str}): Coarse={c_sample}, Fine={f_sample}")
        hierarchy_data = build_single_hierarchy(data, c_sample, f_sample)
        hierarchies.append(hierarchy_data)
            
    return hierarchies


# =============================================================================
# DATASET LOADING (for evaluation scripts)
# =============================================================================

def load_dataset(name: str, root: str = "/tmp"):
    """
    Load a dataset by name.
    
    Args:
        name: 'cora', 'arxiv', or 'mag'
        root: Root directory for data cache
        
    Returns:
        PyG Data object (homogeneous)
    """
    name = name.lower()
    
    if name == 'cora':
        from torch_geometric.datasets import CoraFull
        dataset = CoraFull(root=f"{root}/Cora")
        data = dataset[0]
        # Add standard attributes
        if not hasattr(data, 'node_types'):
            data.node_types = ['paper']
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
        if not hasattr(data, 'global_id'):
            data.global_id = torch.arange(data.num_nodes)
        return data
    
    elif name == 'arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        
        # Fix for PyTorch 2.6+ weights_only default change
        try:
            from torch import serialization as torch_serialization
            from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
            from torch_geometric.data.storage import GlobalStorage
            torch_serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
        except Exception:
            pass  # Older PyTorch versions don't need this
        
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=f"{root}/ogbn_arxiv")
        data = dataset[0]
        if not hasattr(data, 'node_types'):
            data.node_types = ['paper']
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
        if not hasattr(data, 'global_id'):
            data.global_id = torch.arange(data.num_nodes)
        return data
    
    elif name == 'mag':
        from ogb.nodeproppred import PygNodePropPredDataset
        
        # Fix for PyTorch 2.6+ weights_only default change
        try:
            from torch import serialization as torch_serialization
            from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
            from torch_geometric.data.storage import GlobalStorage
            torch_serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
        except Exception:
            pass  # Older PyTorch versions don't need this
        
        dataset = PygNodePropPredDataset(name="ogbn-mag", root=f"{root}/ogbn_mag")
        data = convert_hetero_to_homo(dataset[0])
        return data
    
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: cora, arxiv, mag")


def build_hierarchy(data, num_coarse: int, num_fine: int):
    """
    Build a single hierarchy for evaluation.
    
    Wrapper around build_single_hierarchy with standardized interface.
    
    Args:
        data: PyG Data object
        num_coarse: Number of coarse partitions
        num_fine: Number of fine partitions per coarse
        
    Returns:
        Dict with hierarchy data
    """
    return build_single_hierarchy(data, num_coarse, num_fine)


# =============================================================================
# HIERARCHY CACHING
# =============================================================================

def save_hierarchy(hierarchy: dict, path: str):
    """
    Save hierarchy to pickle file for caching.
    
    Args:
        hierarchy: Dict containing partition data
        path: File path to save to (.pkl)
    """
    import pickle
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(hierarchy, f)
    print(f"[INFO] Saved hierarchy to {path}")


def load_hierarchy(path: str) -> dict:
    """
    Load hierarchy from pickle file.
    
    Args:
        path: File path to load from (.pkl)
        
    Returns:
        Dict with hierarchy data
    """
    import pickle
    with open(path, 'rb') as f:
        hierarchy = pickle.load(f)
    print(f"[INFO] Loaded hierarchy from {path}")
    return hierarchy


def get_or_build_hierarchy(
    data, 
    num_coarse: int, 
    num_fine: int, 
    cache_path: str = None
) -> dict:
    """
    Load cached hierarchy if available, otherwise build and cache.
    
    Args:
        data: PyG Data object
        num_coarse: Number of coarse partitions
        num_fine: Number of fine partitions per coarse
        cache_path: Optional path x hierarchy pickle (e.g., 'cache/cora_hierarchy.pkl')
        
    Returns:
        Dict with hierarchy data
    """
    import os
    
    if cache_path and os.path.exists(cache_path):
        return load_hierarchy(cache_path)
    
    hierarchy = build_hierarchy(data, num_coarse, num_fine)
    
    if cache_path:
        save_hierarchy(hierarchy, cache_path)
    
    return hierarchy
