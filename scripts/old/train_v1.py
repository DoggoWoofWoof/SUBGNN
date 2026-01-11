import modal
import multiprocessing
import sys
import os
import random
import itertools
from collections import Counter, defaultdict

# Fallback for local execution where torch might not be present or partial
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None

# --- HELPER FOR ROBUST PARTITIONING (Global for Multiprocessing) ---

def _partitioner_target(queue, n_parts, xadj, adjncy):
    import pymetis
    try:
        _, membership = pymetis.part_graph(n_parts, xadj=xadj, adjncy=adjncy)
        queue.put(membership)
    except Exception as e:
        print(f"      - [Subprocess] Error during partitioning: {e}", flush=True)
        queue.put(None)

def _extract_fragment_fast_rw(source_graph, target_size):
    from torch_sparse import SparseTensor
    if source_graph.num_nodes == 0 or source_graph.num_edges == 0: return None
    # Assuming source_graph is already on GPU if main process calls this
    device = source_graph.edge_index.device
    
    # On-the-fly sparse tensor creation is fast on GPU
    adj = SparseTensor(row=source_graph.edge_index[0], col=source_graph.edge_index[1], sparse_sizes=(source_graph.num_nodes, source_graph.num_nodes))
    row_ptr, _, _ = adj.csr()
    
    start_node = torch.randint(0, source_graph.num_nodes, (1,), device=device).item()
    
    # random_walk is extremely fast on CUDA
    walk = torch.ops.torch_sparse.random_walk(row_ptr, source_graph.edge_index[1], torch.tensor([start_node], device=device), target_size)[0]
    q_nodes = torch.unique(walk)
    
    if len(q_nodes) < target_size / 2: return None
    return q_nodes

def _extract_subgraph_from_adj(adj_t, node_indices, original_data):
    # Fast subgraph extraction using CSR slicing: O(num_subset_nodes * avg_degree)
    from torch_geometric.data import Data
    
    # 1. Slicing the SparseTensor to get the sub-adjacency
    # adj_t[rows, cols]
    sub_adj = adj_t[node_indices, node_indices]
    
    # 2. Convert back to edge_index (COO format) for PyG Data object
    row, col, _ = sub_adj.coo()
    edge_index = torch.stack([row, col], dim=0)
    
    # 3. Get features (fast indexing)
    x = original_data.x[node_indices]
    
    # 4. Create new Data object
    # Note: node_indices are global IDs. The new edge_index is already re-indexed to 0..len(subset)-1 by the slicing!
    data = Data(x=x, edge_index=edge_index, num_nodes=len(node_indices))
    
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

def generate_multi_coarse_partition_query(original_data, adj_t, coarse_part_graph, fine_graphs, fine_part_nodes_map, fine_to_coarse_map, min_nodes=80, max_nodes=100):
    if coarse_part_graph.number_of_edges() == 0: raise RuntimeError("Coarse graph has no edges.")
    configurations = [(2, 2),(3, 2),(4, 2),(3, 3),(4, 3),(5, 2),(5, 3),(5, 4),(6, 3),(6, 4),(6, 5),(8, 4),(8, 5),(8, 6),(10, 5),(10, 6),(10, 7),(12, 6),(12, 7),(12, 8),(15, 7),(15, 8),(15, 9)]; random.shuffle(configurations)
    
    coarse_to_fine_map = defaultdict(list)
    for f_idx, c_idx in fine_to_coarse_map.items(): coarse_to_fine_map[c_idx].append(f_idx)
    
    for num_frags, min_coarse_parts in configurations:
        possible_start_edges = list(coarse_part_graph.edges()); random.shuffle(possible_start_edges)
        for c_idx1, c_idx2 in possible_start_edges:
            fine_parts_in_c1 = coarse_to_fine_map.get(c_idx1, []); fine_parts_in_c2 = coarse_to_fine_map.get(c_idx2, [])
            all_fine_pairs = list(itertools.product(fine_parts_in_c1, fine_parts_in_c2)); random.shuffle(all_fine_pairs)
            
            for f1, f2 in all_fine_pairs:
                if not are_partitions_neighbors_sparse(adj_t, fine_part_nodes_map[f1], fine_part_nodes_map[f2]): continue
                
                q_fine_indices, queue, visited = [f1, f2], [f1, f2], {f1, f2}
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
                        # local_nodes is relative to fine_graph, fine_part_nodes_map gives global indices
                        all_query_nodes.extend(fine_part_nodes_map[fine_idx][local_nodes].tolist()) # Convert to list for extend
                
                Gq, _ = _finalize_query_from_nodes(original_data, adj_t, all_query_nodes, min_nodes)
                if Gq:
                    stitched_nodes = torch.cat([fine_part_nodes_map[idx] for idx in q_fine_indices])
                    G_stitched = _extract_subgraph_from_adj(adj_t, stitched_nodes, original_data)
                    return Gq, G_stitched, true_coarse_indices
                    
    raise RuntimeError("Failed to generate multi-coarse-partition query.")

def generate_hierarchical_sample(original_data, adj_t, coarse_graphs, fine_graphs, node_to_coarse_tensor, fine_to_coarse_map, fine_part_nodes_map, coarse_part_nodes_map, coarse_part_graph, k=6, q_size_min=20, q_size_max=120, prob_k_hop=0.2, prob_single_part=0.2, prob_multi_coarse=0.4, max_gpos_nodes=4000):
    from torch_geometric.utils import k_hop_subgraph
    rand_choice = random.random(); device = original_data.x.device; Gq, Gpos, G_coarse_pos = None, None, None
    
    if rand_choice < prob_k_hop:
        # k-hop extraction on GPU
        anchor = torch.randint(0, original_data.num_nodes, (1,), device=device).item()
        
        # k_hop_subgraph on GPU
        # subset_pos is a GPU tensor of global node indices
        subset_pos, _, _, _ = k_hop_subgraph(anchor, k, original_data.edge_index, relabel_nodes=False)
        
        if len(subset_pos) < q_size_min or len(subset_pos) > max_gpos_nodes: return None
        
        # O(E_sub) extraction on GPU (SparseTensor slicing)
        Gpos = _extract_subgraph_from_adj(adj_t, subset_pos, original_data)
        
        # Optimized Parent Voting on GPU
        # 1. Map global node indices to coarse partition IDs using the pre-computed GPU tensor
        subset_coarse_ids = node_to_coarse_tensor[subset_pos]
        
        # 2. Filter out invalid mappings (-1) if any (though OGBN-MAG usually fully covered)
        mask = subset_coarse_ids >= 0
        if mask.sum() == 0: return None
        valid_coarse_ids = subset_coarse_ids[mask]
        
        # 3. Find the most frequent coarse ID (mode)
        # torch.mode returns (values, indices). We want the value.
        mode_result = torch.mode(valid_coarse_ids)
        coarse_parent_idx = mode_result.values.item()
        
        if coarse_parent_idx is None: return None
        
        # Sampling within Gpos
        q_nodes_local_indices = torch.randperm(Gpos.num_nodes, device=device)[:random.randint(q_size_min, min(q_size_max, Gpos.num_nodes))]
        
        q_mask = torch.zeros(Gpos.num_nodes, dtype=torch.bool, device=device); q_mask[q_nodes_local_indices] = True
        Gq = Gpos.subgraph(q_mask)
        G_coarse_pos = coarse_graphs[coarse_parent_idx]
        
    elif rand_choice < prob_k_hop + prob_single_part:
        if not fine_graphs: return None
        fine_idx = random.choice(list(fine_to_coarse_map.keys())); Gpos = fine_graphs[fine_idx]
        if Gpos.num_nodes > max_gpos_nodes: return None
        
        # Already on GPU? fine_graphs should be in memory (CPU or GPU, let's ensure they are compatible)
        # Ideally, fine_graphs reside on the same device.
        
        q_nodes_local = _extract_fragment_fast_rw(Gpos, random.randint(q_size_min, q_size_max))
        if q_nodes_local is None: return None
        
        q_mask = torch.zeros(Gpos.num_nodes, dtype=torch.bool, device=device); q_mask[q_nodes_local] = True
        Gq = Gpos.subgraph(q_mask)
        
        coarse_parent_idx = fine_to_coarse_map.get(fine_idx)
        if coarse_parent_idx is None: return None
        G_coarse_pos = coarse_graphs[coarse_parent_idx]
        
    elif rand_choice < prob_k_hop + prob_single_part + prob_multi_coarse:
        try:
            Gq, Gpos, coarse_indices = generate_multi_coarse_partition_query(original_data, adj_t, coarse_part_graph, fine_graphs, fine_part_nodes_map, fine_to_coarse_map, min_nodes=q_size_min, max_nodes=q_size_max)
            all_coarse_pos_nodes = torch.cat([coarse_part_nodes_map[c_idx] for c_idx in coarse_indices])
            G_coarse_pos = _extract_subgraph_from_adj(adj_t, all_coarse_pos_nodes, original_data)
        except RuntimeError: return None
        
    else:
        if not fine_part_nodes_map or len(fine_part_nodes_map) < 2: return None
        num_frags = random.randint(2, 3); start_fine_idx = random.choice(list(fine_part_nodes_map.keys())); coarse_parent_idx = fine_to_coarse_map.get(start_fine_idx)
        if coarse_parent_idx is None: return None
        
        siblings = [idx for idx, c_idx in fine_to_coarse_map.items() if c_idx == coarse_parent_idx]
        source_part_indices = {start_fine_idx}; queue = [start_fine_idx]
        while queue and len(source_part_indices) < num_frags:
            curr_idx = queue.pop(0); random.shuffle(siblings)
            for neighbor_idx in siblings:
                if neighbor_idx not in source_part_indices and are_partitions_neighbors_sparse(adj_t, fine_part_nodes_map[curr_idx], fine_part_nodes_map[neighbor_idx]):
                    source_part_indices.add(neighbor_idx); queue.append(neighbor_idx)
                    if len(source_part_indices) >= num_frags: break
        if len(source_part_indices) < num_frags: return None
        
        pos_nodes = torch.cat([fine_part_nodes_map[i] for i in source_part_indices])
        if len(pos_nodes) > max_gpos_nodes: return None
        
        Gpos = _extract_subgraph_from_adj(adj_t, pos_nodes, original_data)
        
        nodes_per_frag = (q_size_min + q_size_max) // (2 * num_frags); all_query_global_nodes = []
        for fine_idx in source_part_indices:
            local_indices = _extract_fragment_fast_rw(fine_graphs[fine_idx], nodes_per_frag)
            if local_indices is None: return None
            all_query_global_nodes.extend(fine_part_nodes_map[fine_idx][local_indices].tolist())
            
        Gq, _ = _finalize_query_from_nodes(original_data, adj_t, all_query_global_nodes, q_size_min)
        if Gq is None: return None
        G_coarse_pos = coarse_graphs[coarse_parent_idx]
        
    if Gq is None or Gpos is None or G_coarse_pos is None: return None
    return Gq, Gpos, G_coarse_pos

class JigsawDataset(Dataset):
    def __init__(self, original_data, adj_t, hierarchies, batch_size, steps_per_epoch):
        self.original_data = original_data
        self.adj_t = adj_t # Full GPU SparseTensor
        self.hierarchies = hierarchies
        
        # Optimize: Pre-convert node_to_coarse_map to GPU tensor for each hierarchy
        self.node_to_coarse_tensors = []
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
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
    
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        # Everything happens in the same process, directly on GPU tensors
        # Loop until we get a valid sample (failed samples return None)
        while True:
            # Randomly select a hierarchy each time we retry
            h_idx = random.randint(0, len(self.hierarchies) - 1)
            h_data = self.hierarchies[h_idx]
            node_mapper = self.node_to_coarse_tensors[h_idx]
            
            # Unpack hierarchy data
            try:
                sample = generate_hierarchical_sample(
                    self.original_data, self.adj_t, 
                    h_data['coarse_graphs'], h_data['fine_graphs'], 
                    node_mapper, # Passing Tensor instead of dict
                    h_data['fine_to_coarse_map'],
                    h_data['fine_part_nodes_map'], h_data['coarse_part_nodes_map'],
                    h_data['coarse_part_graph']
                )
                if sample:
                    return sample
            except RuntimeError:
                continue # Retry on error

def jigsaw_collate_fn(batch_list):
    from torch_geometric.data import Batch
    gqs = [b[0] for b in batch_list]
    gpos = [b[1] for b in batch_list]
    gcs = [b[2] for b in batch_list]
    return Batch.from_data_list(gqs), Batch.from_data_list(gpos), Batch.from_data_list(gcs)

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
)

app = modal.App("jigsaw-mag-training-full-graph", image=image)

@app.function(gpu="a100", timeout=32400, cpu=8.0, memory=32768)
def train(epochs, steps_per_epoch, batch_size, num_hierarchies=1):
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
    
    # Configure stdout/stderr buffering
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.data import Batch, Data, HeteroData
    from torch_geometric.nn import GINConv, global_mean_pool, GATConv, global_max_pool, global_add_pool
    from torch_sparse import SparseTensor

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
        Convert OGBN-MAG HeteroData -> homogeneous Data
        """
        print("  - Converting heterogeneous graph to homogeneous...")
        node_types = list(hetero_data.num_nodes_dict.keys())
        node_offset, total_nodes = {}, 0
        for nt in node_types:
            node_offset[nt] = total_nodes
            total_nodes += hetero_data.num_nodes_dict[nt]

        feat_dim = hetero_data.x_dict["paper"].size(1)
        x = torch.zeros(total_nodes, feat_dim, dtype=torch.float)
        p_start, p_end = node_offset["paper"], node_offset["paper"] + hetero_data.num_nodes_dict["paper"]
        x[p_start:p_end] = hetero_data.x_dict["paper"]

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

    class NodeFeatureAugmentor(nn.Module):
        def __init__(self, num_nodes: int, num_types: int, type_dim: int = 16, node_dim: int = 0):
            super().__init__(); self.type_emb = nn.Embedding(num_types, type_dim); self.node_dim = node_dim
            self.node_emb = nn.Embedding(num_nodes, node_dim) if node_dim > 0 else None
        @property
        def added_dim(self) -> int: return self.type_emb.embedding_dim + (self.node_emb.embedding_dim if self.node_emb is not None else 0)
        def forward(self, data: Data) -> torch.Tensor:
            pieces = [data.x, self.type_emb(data.node_type)]
            if self.node_emb is not None:
                gid = data.global_id if hasattr(data, "global_id") else torch.arange(data.num_nodes, device=data.x.device)
                pieces.append(self.node_emb(gid))
            return torch.cat(pieces, dim=1)

    def make_undirected_fast(edge_index, num_nodes):
        # This part runs on CPU initially or we can move edge_index to GPU first
        # Since we are immediately moving to GPU after, let's keep this as is for robust conversion
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to_symmetric()
        row, col, _ = adj.coo()
        return torch.stack([row, col], dim=0)

    # --- MODEL ARCHITECTURE ---
    class ImprovedSubgraphEncoder(torch.nn.Module):
        def __init__(self, in_neurons, hidden_neurons, output_neurons, dropout=0.1, use_residual=True, use_attention=False):
            super().__init__()
            self.use_residual = use_residual
            self.use_attention = use_attention
            self.dropout = dropout

            if not use_attention:
                nn1 = Sequential(Linear(in_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
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
            self.input_proj = Linear(in_neurons, hidden_neurons) if in_neurons != hidden_neurons else None
            self.use_multi_pool = True; readout_dim = hidden_neurons * 6 * 3

            self.readout_proj = Sequential(
                Linear(readout_dim, hidden_neurons * 2), ReLU(), Dropout(dropout),
                Linear(hidden_neurons * 2, hidden_neurons), ReLU(), Dropout(dropout),
                Linear(hidden_neurons, output_neurons)
            )
            self.readout_skip = Linear(readout_dim, output_neurons)

        def forward(self, x, edge_index, batch):
            layer_outputs = []
            x_res = self.input_proj(x) if self.input_proj is not None else x

            h1 = F.relu(self.ln1(self.conv1(x, edge_index) + (x_res if self.use_residual and self.conv1(x, edge_index).shape == x_res.shape else 0)))
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
            return F.normalize(self.readout_proj(h_final) + self.readout_skip(h_final), dim=1)

    # --- HIERARCHICAL LOSS ---
    def info_nce_loss(queries, positives, temperature=0.1):
        logits = torch.matmul(queries, positives.T) / temperature; labels = torch.arange(len(queries), device=queries.device)
        return F.cross_entropy(logits, labels)
    def hierarchical_info_nce_loss(zq, z_fine, z_coarse, temperature=0.1, alpha=0.5):
        loss_fine = info_nce_loss(zq, z_fine, temperature); loss_coarse = info_nce_loss(zq, z_coarse, temperature)
        return (alpha * loss_fine) + ((1 - alpha) * loss_coarse)

    # --- DATA PARTITIONING AND HIERARCHY HELPERS ---
    def make_partitions(dataset, num_parts):
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
        if num_parts <= 1: return [dataset], {0: torch.arange(dataset.num_nodes, device=dataset.x.device)}
        
        # Partitioning needs to happen on CPU (pymetis requirement usually)
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
                
                nodes_tensor = torch.tensor(node_indices, dtype=torch.long, device=dataset.x.device)
                part_nodes_map[part_id] = nodes_tensor
                
                # Manual Data construction to avoid implicit subgraph issues and ensure correct relabeling
                relabeled_edge_index, _ = subgraph(nodes_tensor, dataset.edge_index, relabel_nodes=True, num_nodes=dataset.num_nodes)
                
                # Verify relabeled edge index
                # if relabeled_edge_index.numel() > 0 and relabeled_edge_index.max() >= len(node_indices):
                #      raise RuntimeError(f"Relabeled edge index OOB: max {relabeled_edge_index.max()} >= num_sub_nodes {len(node_indices)}")
                
                part_data = Data(edge_index=relabeled_edge_index, num_nodes=len(nodes_tensor))
                
                # Copy attributes manually to be safe
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

                # Ensure global_id is present (required for consistency with _extract_subgraph_from_adj)
                if hasattr(dataset, 'global_id') and dataset.global_id is not None:
                    part_data.global_id = dataset.global_id[nodes_tensor]
                else:
                    # If global_id is missing, use the indices into the current dataset (which might be global)
                    part_data.global_id = nodes_tensor
                
                part_graphs.append(part_data)
        return part_graphs, part_nodes_map

    def build_single_hierarchy(data, num_coarse, num_fine):
        print(f"\n  • Building hierarchy with {num_coarse} coarse partitions...")
        coarse_graphs, coarse_part_nodes_map = make_partitions(data, num_coarse)
        
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
        
        fine_graphs, fine_part_nodes_map, fine_to_coarse_map = [], {}, {}; fine_global_idx = 0
        iterator = tqdm(enumerate(coarse_graphs), total=len(coarse_graphs), desc="    - Creating fine partitions", unit="coarse_part", ncols=100, mininterval=30.0)
        for coarse_idx, coarse_graph in iterator:
            if coarse_idx not in coarse_part_nodes_map: continue
            global_nodes_of_this_coarse_part = coarse_part_nodes_map[coarse_idx]
            if coarse_graph.num_nodes < (num_fine * 2) or coarse_graph.num_edges == 0:
                finer_partitions, finer_nodes_map_local = [coarse_graph], {0: torch.arange(coarse_graph.num_nodes, device=data.x.device)}
            else: finer_partitions, finer_nodes_map_local = make_partitions(coarse_graph, num_fine)
            for fine_local_idx, fine_part in enumerate(finer_partitions):
                if fine_local_idx not in finer_nodes_map_local: continue
                local_indices_in_coarse = finer_nodes_map_local[fine_local_idx]
                global_indices_for_fine = global_nodes_of_this_coarse_part[local_indices_in_coarse]
                if fine_part.num_nodes > 10 and fine_part.num_edges > 0:
                    fine_graphs.append(fine_part); fine_part_nodes_map[fine_global_idx] = global_indices_for_fine
                    fine_to_coarse_map[fine_global_idx] = coarse_idx; fine_global_idx += 1
        return {
            'coarse_graphs': coarse_graphs,
            'fine_graphs': fine_graphs,
            'node_to_coarse_map': node_to_coarse_map,
            'fine_to_coarse_map': fine_to_coarse_map,
            'fine_part_nodes_map': fine_part_nodes_map,
            'coarse_part_graph': coarse_part_graph,
            'coarse_part_nodes_map': coarse_part_nodes_map
        }

    def build_multiple_hierarchies(data, n_hierarchies):
        print(f"[SETUP] Building {n_hierarchies} different hierarchies for Jigsaw training...")
        hierarchies = []; iterator = tqdm(range(n_hierarchies), desc="Building hierarchies", unit="hierarchy", mininterval=30.0)
        for i in iterator:
            num_coarse = random.randint(1900, 2000)
            num_fine = random.randint(5, 10)
            hierarchy_data = build_single_hierarchy(data, num_coarse, num_fine)
            hierarchies.append(hierarchy_data)
        return hierarchies

    # --- CORE TRAINING LOGIC ---
    device = torch.device("cuda"); print(f"[REMOTE INFO] Using device: {device}", flush=True)
    print("[REMOTE INFO] Loading OGBN-MAG (heterogeneous)...", flush=True)
    dataset = PygNodePropPredDataset(name="ogbn-mag", root="/tmp/ogbn_mag_data")
    data = convert_hetero_to_homo(dataset[0])
    
    # Initialize global_id attribute (essential for tracking nodes across partitions)
    if not hasattr(data, 'global_id'):
        data.global_id = torch.arange(data.num_nodes)
        
    print("\n[INFO] Symmetrizing full graph with SparseTensor...", flush=True)
    data.edge_index = make_undirected_fast(data.edge_index, data.num_nodes)
    print(f"  - Undirected edges: {data.edge_index.size(1)}", flush=True)

    print(f"[INFO] Moving entire graph to GPU: {device}...", flush=True)
    data = data.to(device)
    print("  - Graph is on GPU.", flush=True)

    TYPE_DIM = 16; NODE_DIM = 16
    augmentor = NodeFeatureAugmentor(num_nodes=data.num_nodes, num_types=len(data.node_types), type_dim=TYPE_DIM, node_dim=NODE_DIM).to(device)
    base_feat_dim = data.x.size(1); augmented_feat_dim = base_feat_dim + augmentor.added_dim
    print(f"\n[INFO] Base features: {base_feat_dim}, Augmented features: {augmented_feat_dim}", flush=True)

    # --- OPTIMIZATION: USE SPARSETENSOR INSTEAD OF DICT ---
    print("[SETUP] Building SparseTensor adjacency for efficient slicing (on GPU)...", flush=True)
    # create sparse tensor and move to GPU
    adj_t = SparseTensor(
        row=data.edge_index[0], 
        col=data.edge_index[1], 
        sparse_sizes=(data.num_nodes, data.num_nodes)
    ).to(device)
    # Pre-process CSR for fast lookup
    adj_t.csr() 
    print("  - SparseTensor built and on GPU.", flush=True)

    encoder = ImprovedSubgraphEncoder(augmented_feat_dim, 256, 128, use_attention=False, dropout=0.1, use_residual=True).to(device)
    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), augmentor.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)
    encoder.train(); augmentor.train()
    
    # Hierarchy building involves partitioning which might use CPU/METIS, but the resulting subgraphs will be on GPU.
    hierarchies = build_multiple_hierarchies(data, num_hierarchies)
    print("-" * 50)

    for epoch in range(epochs):
        total_loss = 0
        total_samples = steps_per_epoch * batch_size
        
        # Dataset holds GPU references
        dataset_obj = JigsawDataset(data, adj_t, hierarchies, batch_size, total_samples)
        
        # Num_workers=0 ensures main process (GPU access) handles sampling
        loader = DataLoader(
            dataset_obj, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0, 
            collate_fn=jigsaw_collate_fn 
        )

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="step", mininterval=10.0)
        
        for step, (query_batch, pos_batch, coarse_pos_batch) in enumerate(pbar):
            optimizer.zero_grad()
            try:
                # Batches are already on GPU
                xq = augmentor(query_batch); xp = augmentor(pos_batch); xc = augmentor(coarse_pos_batch)
                
                zq = encoder(xq, query_batch.edge_index, query_batch.batch)
                z_pos = encoder(xp, pos_batch.edge_index, pos_batch.batch)
                z_coarse = encoder(xc, coarse_pos_batch.edge_index, coarse_pos_batch.batch)
                
                loss = hierarchical_info_nce_loss(zq, z_pos, z_coarse)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(augmentor.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item(); pbar.set_postfix({"loss": loss.item()})
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"WARNING: OOM at step {step}. Skipping batch.")
                    torch.cuda.empty_cache(); continue
                else: raise e
            # No explicit del needed usually, but good for safety
            del query_batch, pos_batch, coarse_pos_batch, xq, xp, xc, zq, z_pos, z_coarse, loss

        avg_loss = total_loss / steps_per_epoch if steps_per_epoch > 0 else 0
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.6f}, LR = {current_lr:.1e}")

    print("\n[REMOTE INFO] Training finished.")
    return {'encoder': encoder.cpu().state_dict(), 'augmentor': augmentor.cpu().state_dict()}

# --- THE LOCAL ENTRYPOINT ---

@app.local_entrypoint()
def main():
    import torch
    print("🚀 Starting Jigsaw GNN training on Modal for OGBN-MAG...")
    model_state_dicts = train.remote(
        epochs=10,
        steps_per_epoch=50,
        batch_size=64,
        num_hierarchies=1
    )
    file_path = "mag-6_layer-model-jigsaw.pth"
    torch.save(model_state_dicts, file_path)
    print(f"✅ Model saved to '{file_path}'")