"""
Glasgow Solver Benchmark Script

Benchmark: Glasgow on Stitch (FAISS-guided) vs Glasgow on Full Graph
 Datasets: CoraFull, ogbn-arxiv, OGBN-MAG, PubMed, CiteSeer
Timeout: 3 minutes per solver call
Queries: 100 per dataset (distributed across query types)

Usage:
    python scripts/benchmark_glasgow.py                          # All datasets
    python scripts/benchmark_glasgow.py --dataset citeseer       # Single dataset
    python scripts/benchmark_glasgow.py --dataset arxiv --queries 50
"""

import argparse
import csv
import gc
import os
import sys
import time
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Add parent dir so 'src' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import faiss
from collections import defaultdict, deque
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_sparse import SparseTensor

from src.model import ImprovedSubgraphEncoder, RelationAwareSubgraphEncoder, get_graph_embedding
from src.data import make_partitions, make_undirected_fast, make_undirected_with_edge_type
from src.glasgow_solver import glasgow_solve
from src.utils import are_partitions_neighbors, feature_to_label
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx, k_hop_subgraph

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GIN_HIDDEN = 256
GIN_OUTPUT = 128
SOLVER_TIMEOUT = 60.0  # 1 minute
FAISS_TOP_K = 20
FAISS_SCORE_K = 0
QUERY_GENERATOR_VERSION = "aligned_connected_v3"
STITCH_STRATEGY = "ranked"
STITCH_LEVELS = None
STITCH_SEED_COUNT = 20
BOUNDARY_EXPAND_COARSE_BUDGET = 0
MC_DROPOUT_PASSES = 0
MC_DROPOUT_TOP_K = 20
REQUIRE_CANDIDATE_FULLCOV = False
PRUNE_TARGET_BY_QUERY_LABELS = False

# Glasgow binary path (set via env or default)
GLASGOW_BIN = os.environ.get(
    'GLASGOW_SOLVER_BIN',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'bin', 'glasgow_subgraph_solver'),
)

# Dataset configs: (coarse_partitions, fine_per_coarse, model_path, in_features)
DATASET_CONFIGS = {
    'corafull': {
        'coarse': 20,
        'fine': 5,
        'model': 'models/cora-6_layer-model-jigsaw.pth',
        'loader': 'corafull',
    },
    'arxiv': {
        'coarse': 200,
        'fine': 5,
        'model': 'models/arxiv-6_layer-model-jigsaw_v3.pth',
        'loader': 'arxiv',
    },
    'pubmed': {
        'coarse': 20,
        'fine': 5,
        'model': 'models/pubmed-6_layer-model-jigsaw.pth',
        'loader': 'pubmed',
    },
    'citeseer': {
        'coarse': 10,
        'fine': 5,
        'model': 'models/citeseer-6_layer-model-jigsaw.pth',
        'loader': 'citeseer',
    },
    'physics': {
        'coarse': 35,
        'fine': 5,
        'model': 'models/physics-6_layer-model-jigsaw.pth',
        'loader': 'physics',
    },
    'mag': {
        'coarse': 500,
        'fine': 10,
        'model': 'models/mag-6_layer-model-jigsaw.pth',
        'loader': 'mag',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(name):
    """Load dataset and return PyG Data object."""
    if name == 'corafull':
        from torch_geometric.datasets import CoraFull
        dataset = CoraFull(root='data/CoraFull')
        data = dataset[0]
    elif name == 'arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/ogbn_arxiv')
        data = dataset[0]
    elif name == 'pubmed':
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root='data/PubMed', name='PubMed')
        data = dataset[0]
    elif name == 'citeseer':
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root='data/CiteSeer', name='CiteSeer')
        data = dataset[0]
    elif name == 'physics':
        from torch_geometric.datasets import Coauthor
        dataset = Coauthor(root='data/CoauthorPhysics', name='Physics')
        data = dataset[0]
    elif name == 'mag':
        from ogb.nodeproppred import PygNodePropPredDataset
        from src.data import convert_hetero_to_homo
        dataset = PygNodePropPredDataset(name='ogbn-mag', root='data/ogbn_mag')
        data = convert_hetero_to_homo(dataset[0])
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Standardize
    if not hasattr(data, 'global_id') or data.global_id is None:
        data.global_id = torch.arange(data.num_nodes, dtype=torch.long)

    # Make undirected
    print(f"  Symmetrizing edges...", flush=True)
    if hasattr(data, "edge_type") and data.edge_type is not None:
        data.edge_index, data.edge_type, data.num_edge_types = make_undirected_with_edge_type(
            data.edge_index, data.edge_type, data.num_nodes
        )
        print(f"  Preserved relation ids: {data.num_edge_types}", flush=True)
    else:
        data.edge_index = make_undirected_fast(data.edge_index, data.num_nodes)
        data.num_edge_types = 1

    print(f"  Loaded: {data.num_nodes:,} nodes, {data.edge_index.size(1):,} edges, "
          f"{data.x.size(1)} features", flush=True)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class GraphSAGEPartitionEncoder(torch.nn.Module):
    """Matched GraphSAGE baseline with the same readout/projection as Jigsaw."""

    def __init__(
        self,
        in_neurons,
        hidden_neurons=GIN_HIDDEN,
        output_neurons=GIN_OUTPUT,
        dropout=0.1,
        use_residual=True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.input_proj = torch.nn.Linear(in_neurons, hidden_neurons)
        self.conv1 = SAGEConv(hidden_neurons, hidden_neurons)
        self.conv2 = SAGEConv(hidden_neurons, hidden_neurons)
        self.conv3 = SAGEConv(hidden_neurons, hidden_neurons)
        self.conv4 = SAGEConv(hidden_neurons, hidden_neurons)
        self.conv5 = SAGEConv(hidden_neurons, hidden_neurons)
        self.conv6 = SAGEConv(hidden_neurons, hidden_neurons)
        self.ln1 = torch.nn.LayerNorm(hidden_neurons)
        self.ln2 = torch.nn.LayerNorm(hidden_neurons)
        self.ln3 = torch.nn.LayerNorm(hidden_neurons)
        self.ln4 = torch.nn.LayerNorm(hidden_neurons)
        self.ln5 = torch.nn.LayerNorm(hidden_neurons)
        self.ln6 = torch.nn.LayerNorm(hidden_neurons)
        readout_dim = hidden_neurons * 6 * 3
        self.readout_proj = torch.nn.Sequential(
            torch.nn.Linear(readout_dim, hidden_neurons * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_neurons * 2, hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_neurons, output_neurons),
        )
        self.readout_skip = torch.nn.Linear(readout_dim, output_neurons)

    def forward(self, x, edge_index, batch):
        feat = x.x if hasattr(x, "x") else x
        feat = torch.nn.functional.relu(self.input_proj(feat))
        layer_outputs = []
        for conv, norm in (
            (self.conv1, self.ln1),
            (self.conv2, self.ln2),
            (self.conv3, self.ln3),
            (self.conv4, self.ln4),
            (self.conv5, self.ln5),
            (self.conv6, self.ln6),
        ):
            updated = conv(feat, edge_index)
            if self.use_residual:
                updated = updated + feat
            feat = torch.nn.functional.relu(norm(updated))
            layer_outputs.append(feat)
        pooled = []
        for layer_out in layer_outputs:
            pooled.extend(
                [
                    global_mean_pool(layer_out, batch),
                    global_max_pool(layer_out, batch),
                    global_add_pool(layer_out, batch),
                ]
            )
        readout = torch.cat(pooled, dim=1)
        graph_emb = torch.nn.functional.normalize(
            self.readout_proj(readout) + self.readout_skip(readout), dim=1
        )
        return graph_emb, torch.nn.functional.normalize(feat, dim=1)


def _extract_encoder_state(checkpoint):
    if isinstance(checkpoint, dict):
        if "encoder" in checkpoint:
            return checkpoint["encoder"]
        if "encoder_state_dict" in checkpoint:
            return checkpoint["encoder_state_dict"]
    return checkpoint


def _is_graphsage_state(state_dict):
    return any(".lin_l." in key or ".lin_r." in key for key in state_dict)


def _is_rgcn_state(state_dict):
    return any(key.endswith(".root") or key.endswith(".comp") for key in state_dict)


def _infer_rgcn_relations(state_dict):
    comps = [value for key, value in state_dict.items() if key.endswith(".comp") and hasattr(value, "shape")]
    if comps:
        return int(comps[0].shape[0])
    weights = [value for key, value in state_dict.items() if key.endswith(".weight") and hasattr(value, "shape") and len(value.shape) == 3]
    if weights:
        return int(weights[0].shape[0])
    return 1


def load_model(model_path, in_features, device):
    """Load trained encoder and return (encoder, load_time)."""
    t0 = time.time()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = _extract_encoder_state(checkpoint)
        if _is_graphsage_state(state_dict):
            encoder = GraphSAGEPartitionEncoder(in_neurons=in_features).to(device)
            architecture = "GraphSAGE"
        elif _is_rgcn_state(state_dict):
            encoder = RelationAwareSubgraphEncoder(
                in_neurons=in_features,
                hidden_neurons=GIN_HIDDEN,
                output_neurons=GIN_OUTPUT,
                num_relations=_infer_rgcn_relations(state_dict),
            ).to(device)
            architecture = "RGCN"
        else:
            encoder = ImprovedSubgraphEncoder(
                in_neurons=in_features,
                hidden_neurons=GIN_HIDDEN,
                output_neurons=GIN_OUTPUT,
            ).to(device)
            architecture = "GIN"
        encoder.load_state_dict(state_dict)
        print(f"  Loaded {architecture} model from {model_path}", flush=True)
    else:
        encoder = ImprovedSubgraphEncoder(
            in_neurons=in_features,
            hidden_neurons=GIN_HIDDEN,
            output_neurons=GIN_OUTPUT,
        ).to(device)
        print(f"  WARNING: Model not found at {model_path}, using random weights!", flush=True)

    encoder.eval()
    load_time = time.time() - t0
    return encoder, load_time


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHY BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_hierarchy(data, num_coarse, num_fine):
    """Build coarse/fine partition hierarchy. Returns (hierarchy_dict, partition_time)."""
    import networkx as nx

    t0 = time.time()
    print(f"  Building hierarchy: {num_coarse} coarse, {num_fine} fine per coarse...", flush=True)

    coarse_graphs, coarse_part_nodes_map = make_partitions(data, num_coarse, keep_features=False)

    # Build node_to_coarse_map
    node_to_coarse_map = {}
    for cid, nodes in coarse_part_nodes_map.items():
        for nid in nodes.tolist():
            node_to_coarse_map[nid] = cid

    # Build coarse partition graph
    coarse_part_graph = nx.Graph()
    coarse_ids_tensor = torch.full((data.num_nodes,), -1, dtype=torch.long)
    for cid, nodes in coarse_part_nodes_map.items():
        coarse_ids_tensor[nodes] = cid

    src, dst = data.edge_index
    c_src = coarse_ids_tensor[src]
    c_dst = coarse_ids_tensor[dst]
    mask = (c_src != c_dst) & (c_src != -1) & (c_dst != -1)
    edge_lo = torch.minimum(c_src[mask], c_dst[mask])
    edge_hi = torch.maximum(c_src[mask], c_dst[mask])
    c_edges = torch.stack([edge_lo, edge_hi], dim=1)
    c_edges, c_counts = torch.unique(c_edges, dim=0, return_counts=True)
    for (src_c, dst_c), weight in zip(c_edges.cpu().tolist(), c_counts.cpu().tolist()):
        if src_c == dst_c:
            continue
        coarse_part_graph.add_edge(int(src_c), int(dst_c), weight=int(weight))

    # Build fine partitions within each coarse
    fine_graphs = {}
    fine_part_nodes_map = {}
    fine_to_coarse_map = {}
    fine_idx = 0

    for coarse_id, cg in enumerate(tqdm(coarse_graphs, desc="    Fine partitions", ncols=80)):
        if cg is None:
            continue
        actual_cid = getattr(cg, 'part_id', coarse_id)
        if actual_cid not in coarse_part_nodes_map:
            continue

        global_nodes = coarse_part_nodes_map[actual_cid]

        if cg.num_nodes < (num_fine * 2) or cg.num_edges == 0:
            fine_parts = [cg]
            fine_nodes_map_local = {0: torch.arange(cg.num_nodes)}
        else:
            fine_parts, fine_nodes_map_local = make_partitions(cg, num_fine, keep_features=False)

        for flocal_idx, fp in enumerate(fine_parts):
            if flocal_idx not in fine_nodes_map_local:
                continue
            local_indices = fine_nodes_map_local[flocal_idx]
            global_indices = global_nodes[local_indices]
            if fp is not None and global_indices.numel() > 0:
                fine_graphs[fine_idx] = fp
                fine_part_nodes_map[fine_idx] = global_indices
                fine_to_coarse_map[fine_idx] = actual_cid
                fine_idx += 1

    node_to_fine_map = {}
    fine_ids_tensor = torch.full((data.num_nodes,), -1, dtype=torch.long)
    for fid, nodes in fine_part_nodes_map.items():
        nodes_cpu = nodes.detach().cpu().long()
        fine_ids_tensor[nodes_cpu] = int(fid)
        for nid in nodes_cpu.tolist():
            node_to_fine_map[int(nid)] = int(fid)

    fine_part_graph = nx.Graph()
    f_src = fine_ids_tensor[src.detach().cpu().long()]
    f_dst = fine_ids_tensor[dst.detach().cpu().long()]
    f_mask = (f_src != f_dst) & (f_src != -1) & (f_dst != -1)
    if f_mask.any():
        f_lo = torch.minimum(f_src[f_mask], f_dst[f_mask])
        f_hi = torch.maximum(f_src[f_mask], f_dst[f_mask])
        f_edges = torch.stack([f_lo, f_hi], dim=1)
        f_edges, f_counts = torch.unique(f_edges, dim=0, return_counts=True)
        for (src_f, dst_f), weight in zip(f_edges.cpu().tolist(), f_counts.cpu().tolist()):
            if src_f == dst_f:
                continue
            fine_part_graph.add_edge(int(src_f), int(dst_f), weight=int(weight))

    partition_time = time.time() - t0
    print(f"  Hierarchy built: {len(coarse_part_nodes_map)} coarse, "
          f"{len(fine_graphs)} fine partitions ({partition_time:.1f}s)", flush=True)

    return {
        'coarse_graphs': coarse_graphs,
        'coarse_part_nodes_map': coarse_part_nodes_map,
        'coarse_part_graph': coarse_part_graph,
        'node_to_coarse_map': node_to_coarse_map,
        'fine_graphs': fine_graphs,
        'fine_part_nodes_map': fine_part_nodes_map,
        'fine_to_coarse_map': fine_to_coarse_map,
        'fine_part_graph': fine_part_graph,
        'node_to_fine_map': node_to_fine_map,
    }, partition_time


# ═══════════════════════════════════════════════════════════════════════════════
# FAISS INDEX BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(data, hierarchy, encoder, device):
    """Embed all coarse partitions and build FAISS index.
    Returns (faiss_index, faiss_idx_to_coarse_id, build_time).
    """
    t0 = time.time()
    coarse_graphs = hierarchy['coarse_graphs']
    original_data = data

    embeds = []
    faiss_idx_to_coarse_id = {}

    for coarse_id, g in enumerate(tqdm(coarse_graphs, desc="  Building FAISS index", ncols=80)):
        if g is None:
            continue
        g = g.to(device)
        # Fetch features from original data
        if g.x is None:
            if hasattr(g, 'global_id') and g.global_id is not None:
                gids = g.global_id.to(original_data.x.device)
                g.x = original_data.x[gids].to(device)
            else:
                g.x = torch.zeros(g.num_nodes, original_data.x.size(1), device=device)

        emb = get_graph_embedding(g, encoder, device)
        faiss_idx_to_coarse_id[len(embeds)] = coarse_id
        embeds.append(emb)

    coarse_embeds = torch.cat(embeds, dim=0)

    index = faiss.IndexFlatL2(GIN_OUTPUT)
    index.add(coarse_embeds.cpu().numpy())

    build_time = time.time() - t0
    print(f"  FAISS index: {index.ntotal} vectors ({build_time:.1f}s)", flush=True)
    return index, faiss_idx_to_coarse_id, build_time


# ═══════════════════════════════════════════════════════════════════════════════
# SUBGRAPH EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_subgraph(adj_t, node_indices, original_data):
    """Extract subgraph using SparseTensor slicing."""
    if not isinstance(node_indices, torch.Tensor):
        node_indices = torch.tensor(list(node_indices), dtype=torch.long)
    node_indices = node_indices.detach().cpu().long()
    if node_indices.numel() == 0:
        return None
    try:
        sub_adj = adj_t[node_indices, node_indices]
        row, col, value = sub_adj.coo()
        edge_index = torch.stack([row, col], dim=0)
    except Exception:
        return None

    x = original_data.x[node_indices]
    sub = Data(x=x.cpu(), edge_index=edge_index.cpu(), num_nodes=len(node_indices))
    sub.global_id = node_indices.cpu()
    if value is not None:
        sub.edge_type = value.detach().cpu().long()
    for attr in ("node_types", "node_offset", "edge_types", "feature_schema", "num_edge_types"):
        if hasattr(original_data, attr):
            setattr(sub, attr, getattr(original_data, attr))
    for attr in ("node_type", "y", "node_label"):
        value = getattr(original_data, attr, None)
        if isinstance(value, torch.Tensor) and int(value.size(0)) == int(original_data.num_nodes):
            setattr(sub, attr, value[node_indices].detach().cpu())
    return sub


def _subgraph_is_connected(query):
    """Return True when the extracted query is one undirected connected component."""
    if query is None or int(query.num_nodes) <= 0:
        return False
    edge_index = getattr(query, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return int(query.num_nodes) <= 1
    adj = [[] for _ in range(int(query.num_nodes))]
    for u, v in edge_index.detach().cpu().t().tolist():
        u = int(u)
        v = int(v)
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)
    seen = {0}
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return len(seen) == int(query.num_nodes)


def _subgraph_component_sizes(query):
    """Return undirected connected-component sizes for an extracted query."""
    if query is None or int(query.num_nodes) <= 0:
        return []
    n = int(query.num_nodes)
    edge_index = getattr(query, "edge_index", None)
    adj = [[] for _ in range(n)]
    if edge_index is not None and edge_index.numel() > 0:
        for u, v in edge_index.detach().cpu().t().tolist():
            u = int(u)
            v = int(v)
            if u == v:
                continue
            adj[u].append(v)
            adj[v].append(u)
    seen = set()
    sizes = []
    for start in range(n):
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        size = 0
        while queue:
            u = queue.popleft()
            size += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def prune_target_by_query_labels(query_data, stitched_nodes, adj_t, original_data):
    """Keep only candidate target nodes whose Glasgow vertex label appears in the query."""
    if query_data.x is None or original_data.x is None:
        return stitched_nodes
    query_labels = {feature_to_label(query_data.x[i]) for i in range(query_data.num_nodes)}
    kept = []
    for nid in stitched_nodes.detach().cpu().long().tolist():
        label = feature_to_label(original_data.x[int(nid)])
        if label in query_labels:
            kept.append(int(nid))
    if len(kept) < query_data.num_nodes:
        return stitched_nodes
    return torch.tensor(kept, dtype=torch.long)


def _tensor_from_ordered_nodes(nodes):
    """Return a long tensor with duplicate node ids removed while preserving order."""
    seen = set()
    ordered = []
    for n in nodes:
        nid = int(n)
        if nid not in seen:
            seen.add(nid)
            ordered.append(nid)
    return torch.tensor(ordered, dtype=torch.long)


def _split_counts(total, parts, min_each=3):
    """Split a target query size across fragments."""
    if parts <= 0:
        return []
    base = total // parts
    rem = total % parts
    counts = [base + (1 if i < rem else 0) for i in range(parts)]
    return [max(min_each, c) for c in counts]


def _connected_bfs_nodes(adj_t, start_node, target_size, allowed_nodes=None, min_nodes=None):
    """Sample a connected BFS blob, optionally constrained to a node set."""
    min_nodes = min_nodes if min_nodes is not None else max(5, min(target_size, target_size - 10))
    row_ptr, col, _ = adj_t.csr()
    row_ptr = row_ptr.cpu()
    col = col.cpu()

    start = int(start_node)
    allowed = None
    if allowed_nodes is not None:
        if isinstance(allowed_nodes, torch.Tensor):
            allowed = set(int(x) for x in allowed_nodes.detach().cpu().tolist())
        else:
            allowed = set(int(x) for x in allowed_nodes)
        if start not in allowed:
            return None

    out = [start]
    visited = {start}
    queue = deque([start])
    while queue and len(out) < target_size:
        u = queue.popleft()
        if u < 0 or u + 1 >= row_ptr.numel():
            continue
        begin = int(row_ptr[u].item())
        end = int(row_ptr[u + 1].item())
        neighbors = col[begin:end].tolist()
        random.shuffle(neighbors)
        for v in neighbors:
            v = int(v)
            if v in visited:
                continue
            if allowed is not None and v not in allowed:
                continue
            visited.add(v)
            out.append(v)
            queue.append(v)
            if len(out) >= target_size:
                break

    if len(out) < min_nodes:
        return None
    return _tensor_from_ordered_nodes(out[:target_size])


def _sample_partition_fragment(adj_t, part_nodes, target_size, min_nodes=None, attempts=20,
                               required_nodes=None):
    """Sample a connected fragment from an induced partition."""
    if part_nodes is None or len(part_nodes) == 0:
        return None
    part_nodes = part_nodes.detach().cpu().long()
    min_nodes = min_nodes if min_nodes is not None else max(5, min(target_size, target_size - 10))
    if len(part_nodes) < min_nodes:
        return None
    part_set = set(int(x) for x in part_nodes.tolist())
    required = []
    if required_nodes is not None:
        if isinstance(required_nodes, torch.Tensor):
            required = [int(x) for x in required_nodes.detach().cpu().tolist()]
        elif isinstance(required_nodes, (list, tuple, set)):
            required = [int(x) for x in required_nodes]
        else:
            required = [int(required_nodes)]
        required = [x for x in required if x in part_set]
        if required_nodes is not None and not required:
            return None

    starts = required[:]
    for _ in range(attempts):
        starts.append(int(part_nodes[torch.randint(0, len(part_nodes), (1,)).item()].item()))

    required_set = set(required)
    for start_node in starts:
        nodes = _connected_bfs_nodes(
            adj_t,
            int(start_node),
            target_size,
            allowed_nodes=part_nodes,
            min_nodes=min_nodes,
        )
        if nodes is not None and required_set.issubset(set(int(x) for x in nodes.tolist())):
            return nodes
    return None


def _partition_bridge_edge(adj_t, nodes_a, nodes_b):
    """Return one global edge endpoint pair connecting two node sets, if any."""
    if nodes_a is None or nodes_b is None or len(nodes_a) == 0 or len(nodes_b) == 0:
        return None
    nodes_a = nodes_a.detach().cpu().long()
    nodes_b = nodes_b.detach().cpu().long()
    try:
        sub = adj_t[nodes_a, nodes_b]
        row, col, _ = sub.coo()
        if int(row.numel()) > 0:
            return int(nodes_a[int(row[0])].item()), int(nodes_b[int(col[0])].item())
        sub = adj_t[nodes_b, nodes_a]
        row, col, _ = sub.coo()
        if int(row.numel()) > 0:
            return int(nodes_a[int(col[0])].item()), int(nodes_b[int(row[0])].item())
    except Exception:
        nodes_b_set = set(int(x) for x in nodes_b.tolist())
        row_ptr, col, _ = adj_t.csr()
        row_ptr = row_ptr.cpu()
        col = col.cpu()
        for u in nodes_a.tolist():
            u = int(u)
            if u < 0 or u + 1 >= row_ptr.numel():
                continue
            begin = int(row_ptr[u].item())
            end = int(row_ptr[u + 1].item())
            for v in col[begin:end].tolist():
                v = int(v)
                if v in nodes_b_set:
                    return u, v
    return None


def _partitions_touch(adj_t, nodes_a, nodes_b):
    """Return True if two node sets have at least one edge between them."""
    if nodes_a is None or nodes_b is None or len(nodes_a) == 0 or len(nodes_b) == 0:
        return False
    try:
        return adj_t[nodes_a, nodes_b].nnz() > 0 or adj_t[nodes_b, nodes_a].nnz() > 0
    except Exception:
        nodes_b_set = set(int(x) for x in nodes_b.detach().cpu().tolist())
        row_ptr, col, _ = adj_t.csr()
        row_ptr = row_ptr.cpu()
        col = col.cpu()
        for u in nodes_a.detach().cpu().tolist():
            u = int(u)
            if u < 0 or u + 1 >= row_ptr.numel():
                continue
            begin = int(row_ptr[u].item())
            end = int(row_ptr[u + 1].item())
            if any(int(v) in nodes_b_set for v in col[begin:end].tolist()):
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY GENERATION (simplified versions from evaluate.py)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_single_partition_query(data, adj_t, coarse_part_nodes_map, anchor_coarse_idx,
                                    target_size=20):
    """Generate a connected query from a single coarse partition."""
    if anchor_coarse_idx not in coarse_part_nodes_map:
        return None

    nodes = coarse_part_nodes_map[anchor_coarse_idx]
    min_nodes = max(5, target_size - 10)
    if len(nodes) < min_nodes:
        return None

    subset = _sample_partition_fragment(adj_t, nodes, target_size, min_nodes=min_nodes)
    if subset is None:
        return None

    query = extract_subgraph(adj_t, subset, data)
    if query is None or query.num_nodes < 5 or query.num_edges < 2:
        return None

    true_coarse = {anchor_coarse_idx}
    return query, subset, true_coarse, 'single'


def generate_multi_fine_query(data, adj_t, G_nx, fine_part_nodes_map, fine_to_coarse_map,
                              anchor_coarse_idx, target_size=20):
    """Generate connected fragments across 2-4 fine partitions in one coarse partition."""
    candidate_starts = [idx for idx, c in fine_to_coarse_map.items() if c == anchor_coarse_idx]
    if not candidate_starts:
        return None

    for _ in range(50):
        num_frags = random.choice([2, 3, 4])
        if target_size // num_frags < 3:
            continue
        start_fine_idx = random.choice(candidate_starts)
        siblings = list(candidate_starts)

        q_fine_indices = [start_fine_idx]
        visited = {start_fine_idx}
        queue = [start_fine_idx]
        required_by_fine = defaultdict(set)

        while queue and len(q_fine_indices) < num_frags:
            current_idx = queue.pop(0)
            random.shuffle(siblings)
            for neighbor_idx in siblings:
                if neighbor_idx in visited:
                    continue
                edge = _partition_bridge_edge(
                    adj_t,
                    fine_part_nodes_map[current_idx],
                    fine_part_nodes_map[neighbor_idx],
                )
                if edge is None:
                    continue
                required_by_fine[current_idx].add(edge[0])
                required_by_fine[neighbor_idx].add(edge[1])
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)
                q_fine_indices.append(neighbor_idx)
                if len(q_fine_indices) >= num_frags:
                    break

        if len(q_fine_indices) < 2:
            continue

        counts = _split_counts(target_size, len(q_fine_indices), min_each=3)
        all_query_nodes = []
        ok = True
        for fine_idx, frag_size in zip(q_fine_indices, counts):
            part_nodes = fine_part_nodes_map[fine_idx]
            subset = _sample_partition_fragment(
                adj_t,
                part_nodes,
                frag_size,
                min_nodes=max(3, frag_size - 2),
                required_nodes=required_by_fine.get(fine_idx),
            )
            if subset is None:
                ok = False
                break
            all_query_nodes.extend(subset.tolist())
        if not ok:
            continue

        combined = _tensor_from_ordered_nodes(all_query_nodes)
        if len(combined) < max(5, target_size - 10):
            continue
        if len(combined) > target_size:
            combined = combined[:target_size]

        query = extract_subgraph(adj_t, combined, data)
        if query is None or query.num_nodes < 5 or query.num_edges < 2:
            continue
        if not _subgraph_is_connected(query):
            continue

        true_coarse = {anchor_coarse_idx}
        return query, combined, true_coarse, 'multi_fine'

    return None


def generate_multi_coarse_query(data, adj_t, coarse_part_nodes_map, coarse_part_graph,
                                fine_part_nodes_map=None, fine_to_coarse_map=None,
                                target_size=20):
    """Generate connected fine fragments spanning neighboring coarse partitions."""
    edges = list(coarse_part_graph.edges())
    if not edges:
        return None

    if fine_part_nodes_map and fine_to_coarse_map:
        coarse_to_fine = defaultdict(list)
        for fine_idx, coarse_idx in fine_to_coarse_map.items():
            coarse_to_fine[coarse_idx].append(fine_idx)

        configs = [
            (2, 2), (3, 2), (4, 2), (3, 3), (4, 3),
            (5, 2), (5, 3), (5, 4), (6, 3), (6, 4),
        ]
        configs = [cfg for cfg in configs if target_size // cfg[0] >= 3]
        random.shuffle(configs)
        random.shuffle(edges)

        for num_frags, min_coarse_parts in configs:
            for c1, c2 in edges[:50]:
                f1s = list(coarse_to_fine.get(c1, []))
                f2s = list(coarse_to_fine.get(c2, []))
                if not f1s or not f2s:
                    continue
                random.shuffle(f1s)
                random.shuffle(f2s)

                bridge = None
                bridge_edge = None
                for f1 in f1s:
                    for f2 in f2s:
                        edge = _partition_bridge_edge(adj_t, fine_part_nodes_map[f1], fine_part_nodes_map[f2])
                        if edge is not None:
                            bridge = (f1, f2)
                            bridge_edge = edge
                            break
                    if bridge is not None:
                        break
                if bridge is None:
                    continue

                q_fine_indices = [bridge[0], bridge[1]]
                visited = set(q_fine_indices)
                queue = deque(q_fine_indices)
                required_by_fine = {
                    bridge[0]: {bridge_edge[0]},
                    bridge[1]: {bridge_edge[1]},
                }
                while queue and len(q_fine_indices) < num_frags:
                    current = queue.popleft()
                    current_coarse = fine_to_coarse_map[current]
                    neighbor_coarse = list(coarse_part_graph.neighbors(current_coarse)) + [current_coarse]
                    candidate_fines = [
                        f for cid in neighbor_coarse for f in coarse_to_fine.get(cid, [])
                    ]
                    random.shuffle(candidate_fines)
                    for candidate in candidate_fines:
                        if candidate in visited:
                            continue
                        edge = _partition_bridge_edge(
                            adj_t,
                            fine_part_nodes_map[current],
                            fine_part_nodes_map[candidate],
                        )
                        if edge is None:
                            continue
                        required_by_fine.setdefault(current, set()).add(edge[0])
                        required_by_fine.setdefault(candidate, set()).add(edge[1])
                        visited.add(candidate)
                        q_fine_indices.append(candidate)
                        queue.append(candidate)
                        if len(q_fine_indices) >= num_frags:
                            break

                true_coarse = {fine_to_coarse_map[f] for f in q_fine_indices}
                if len(q_fine_indices) < num_frags or len(true_coarse) < min_coarse_parts:
                    continue

                counts = _split_counts(target_size, len(q_fine_indices), min_each=3)
                all_query_nodes = []
                ok = True
                for fine_idx, frag_size in zip(q_fine_indices, counts):
                    subset = _sample_partition_fragment(
                        adj_t,
                        fine_part_nodes_map[fine_idx],
                        frag_size,
                        min_nodes=max(3, frag_size - 2),
                        required_nodes=required_by_fine.get(fine_idx),
                    )
                    if subset is None:
                        ok = False
                        break
                    all_query_nodes.extend(subset.tolist())
                if not ok:
                    continue

                combined = _tensor_from_ordered_nodes(all_query_nodes)
                if len(combined) < max(5, target_size - 10):
                    continue
                if len(combined) > target_size:
                    combined = combined[:target_size]

                query = extract_subgraph(adj_t, combined, data)
                if query is None or query.num_nodes < 5 or query.num_edges < 2:
                    continue
                if not _subgraph_is_connected(query):
                    continue
                return query, combined, true_coarse, 'multi_coarse'

    random.shuffle(edges)
    for c1, c2 in edges[:50]:
        nodes1 = coarse_part_nodes_map.get(c1)
        nodes2 = coarse_part_nodes_map.get(c2)
        if nodes1 is None or nodes2 is None:
            continue

        counts = _split_counts(target_size, 2, min_each=5)
        bridge_edge = _partition_bridge_edge(adj_t, nodes1, nodes2)
        if bridge_edge is None:
            continue

        subset1 = _sample_partition_fragment(
            adj_t,
            nodes1,
            counts[0],
            min_nodes=max(5, counts[0] - 2),
            required_nodes=[bridge_edge[0]],
        )
        subset2 = _sample_partition_fragment(
            adj_t,
            nodes2,
            counts[1],
            min_nodes=max(5, counts[1] - 2),
            required_nodes=[bridge_edge[1]],
        )
        if subset1 is None or subset2 is None:
            continue

        combined = _tensor_from_ordered_nodes(subset1.tolist() + subset2.tolist())
        if len(combined) < max(5, target_size - 10):
            continue
        if len(combined) > target_size:
            combined = combined[:target_size]

        query = extract_subgraph(adj_t, combined, data)
        if query is None or query.num_nodes < 5 or query.num_edges < 2:
            continue
        if not _subgraph_is_connected(query):
            continue

        true_coarse = {c1, c2}
        return query, combined, true_coarse, 'multi_coarse'

    return None


def generate_k_hop_query(data, adj_t, target_size=20):
    """Generate a training-matched connected blob inside a 3-hop neighborhood."""
    min_nodes = max(5, target_size - 10)
    for _ in range(50):
        anchor = random.randint(0, data.num_nodes - 1)
        try:
            subset, _, _, _ = k_hop_subgraph(
                anchor, num_hops=3, edge_index=data.edge_index,
                relabel_nodes=False, num_nodes=data.num_nodes
            )
        except Exception:
            continue
        if len(subset) < min_nodes:
            continue

        query_nodes = _connected_bfs_nodes(
            adj_t,
            anchor,
            target_size,
            allowed_nodes=subset,
            min_nodes=min_nodes,
        )
        if query_nodes is None:
            continue

        query = extract_subgraph(adj_t, query_nodes, data)
        if query is None or query.num_nodes < 5 or query.num_edges < 2:
            continue

        return query, query_nodes, None, 'k_hop'  # true_coarse determined later

    return None


def determine_true_coarse(node_indices, node_to_coarse_map):
    """Determine which coarse partitions a query touches."""
    coarse_ids = set()
    for nid in node_indices.tolist():
        if nid in node_to_coarse_map:
            coarse_ids.add(node_to_coarse_map[nid])
    return coarse_ids


def determine_true_fine(node_indices, node_to_fine_map):
    """Determine which fine partitions a query touches."""
    fine_ids = set()
    for nid in node_indices.tolist():
        if nid in node_to_fine_map:
            fine_ids.add(node_to_fine_map[nid])
    return fine_ids


def _unique_ordered(items):
    """Return items with duplicates removed while preserving order."""
    seen = set()
    out = []
    for item in items:
        item = int(item)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _stitch_levels(faiss_top_k):
    """Candidate sizes to try before giving up on a query."""
    if STITCH_LEVELS:
        raw_levels = STITCH_LEVELS
    else:
        raw_levels = [1, 5, 20, 35, 50, 75, 100, faiss_top_k]
    levels = []
    for level in raw_levels:
        level = int(level)
        if 1 <= level <= faiss_top_k and level not in levels:
            levels.append(level)
    return [(level, f"top-{level}") for level in levels]


def _select_ranked_stitch_ids(ranked_ids, max_parts, coarse_part_graph):
    """Existing rank-first stitch selection with deterministic ordering."""
    ranked_ids = _unique_ordered(ranked_ids)
    top_indices = set(ranked_ids[:max_parts])
    if max_parts > 1 and coarse_part_graph is not None:
        expanded = set(top_indices)
        for cidx in list(top_indices):
            if coarse_part_graph.has_node(cidx):
                expanded.update(int(n) for n in coarse_part_graph.neighbors(cidx))

        stitched_ids = []
        for idx in ranked_ids:
            if idx in expanded:
                stitched_ids.append(idx)
                if len(stitched_ids) >= max_parts:
                    break
        for idx in sorted(expanded):
            if idx not in stitched_ids:
                stitched_ids.append(idx)
                if len(stitched_ids) >= max_parts:
                    break
        return stitched_ids

    return ranked_ids[:max_parts]


def _select_neighbor_rerank_stitch_ids(ranked_ids, max_parts, coarse_part_graph, seed_count):
    """
    Retrieve deep, then stitch compactly.

    Keep the highest-confidence seeds, then fill the candidate set with graph
    neighbors that have strong boundary support to those seeds. This is meant
    for k-hop-style queries where the true coarse partitions should form a
    connected or nearly connected region in the coarse partition graph.
    """
    ranked_ids = _unique_ordered(ranked_ids)
    if coarse_part_graph is None or max_parts <= 1:
        return ranked_ids[:max_parts]

    rank_pos = {cid: pos for pos, cid in enumerate(ranked_ids)}
    fallback_rank = len(ranked_ids) + 50
    selected = []
    selected_set = set()

    def add(cid):
        cid = int(cid)
        if cid in selected_set:
            return False
        if not coarse_part_graph.has_node(cid) and cid not in rank_pos:
            return False
        selected.append(cid)
        selected_set.add(cid)
        return True

    seed_limit = min(max_parts, max(1, min(seed_count, len(ranked_ids))))
    for cid in ranked_ids[:seed_limit]:
        add(cid)

    neighbor_scores = defaultdict(float)
    for seed in selected[:seed_limit]:
        if not coarse_part_graph.has_node(seed):
            continue
        seed_rank_bonus = 1.0 / (1.0 + rank_pos.get(seed, fallback_rank))
        for nb in coarse_part_graph.neighbors(seed):
            nb = int(nb)
            if nb in selected_set:
                continue
            edge_weight = coarse_part_graph[seed][nb].get('weight', 1.0)
            nb_rank_bonus = 1.0 / (1.0 + rank_pos.get(nb, fallback_rank))
            neighbor_scores[nb] += np.log1p(float(edge_weight)) * (1.0 + seed_rank_bonus) + nb_rank_bonus

    for cid, _ in sorted(neighbor_scores.items(), key=lambda kv: (-kv[1], rank_pos.get(kv[0], fallback_rank), kv[0])):
        add(cid)
        if len(selected) >= max_parts:
            return selected

    for cid in ranked_ids:
        add(cid)
        if len(selected) >= max_parts:
            return selected

    return selected


def _select_coarse_boundary_expand_ids(seed_ranked_ids, max_parts, coarse_part_graph,
                                       seed_count, score_ranked_ids=None):
    """
    Start from a small FAISS seed set, then expand through the coarse boundary.

    This is the "dynamic retrieval" path: the seed set can be top-20, while the
    final coarse candidate budget can be 50/75/100/etc. New partitions are chosen
    from graph frontier support instead of blindly taking deeper FAISS ranks.
    If a larger score-ranked list is supplied, it is used only as a tie-breaker.
    """
    seed_ranked_ids = _unique_ordered(seed_ranked_ids)
    score_ranked_ids = _unique_ordered(score_ranked_ids or seed_ranked_ids)
    if coarse_part_graph is None or max_parts <= 1:
        return seed_ranked_ids[:max_parts]

    rank_pos = {cid: pos for pos, cid in enumerate(score_ranked_ids)}
    fallback_rank = len(score_ranked_ids) + 50
    selected = []
    selected_set = set()
    frontier_scores = defaultdict(float)
    frontier_support = defaultdict(int)

    def add_frontier_from(cid):
        if not coarse_part_graph.has_node(cid):
            return
        seed_rank_bonus = 1.0 / (1.0 + rank_pos.get(cid, fallback_rank))
        for nb in coarse_part_graph.neighbors(cid):
            nb = int(nb)
            if nb in selected_set:
                continue
            edge_weight = coarse_part_graph[cid][nb].get('weight', 1.0)
            nb_rank_bonus = 1.0 / (1.0 + rank_pos.get(nb, fallback_rank))
            frontier_scores[nb] += np.log1p(float(edge_weight)) * (1.0 + seed_rank_bonus) + nb_rank_bonus
            frontier_support[nb] += 1

    def add(cid):
        cid = int(cid)
        if cid in selected_set:
            return False
        if not coarse_part_graph.has_node(cid) and cid not in rank_pos:
            return False
        selected.append(cid)
        selected_set.add(cid)
        frontier_scores.pop(cid, None)
        frontier_support.pop(cid, None)
        add_frontier_from(cid)
        return True

    default_seed_count = len(seed_ranked_ids)
    seed_limit = min(max_parts, max(1, min(seed_count or default_seed_count, len(seed_ranked_ids))))
    for cid in seed_ranked_ids[:seed_limit]:
        add(cid)

    while len(selected) < max_parts and frontier_scores:
        best = min(
            frontier_scores,
            key=lambda cid: (
                -frontier_scores[cid],
                -frontier_support[cid],
                rank_pos.get(cid, fallback_rank),
                cid,
            ),
        )
        add(best)

    for cid in score_ranked_ids:
        add(cid)
        if len(selected) >= max_parts:
            return selected

    return selected


def _select_stitch_ids(ranked_ids, max_parts, coarse_part_graph, score_ranked_ids=None):
    if STITCH_STRATEGY == "coarse_boundary_expand":
        budget = BOUNDARY_EXPAND_COARSE_BUDGET if BOUNDARY_EXPAND_COARSE_BUDGET > 0 else max_parts
        return _select_coarse_boundary_expand_ids(
            ranked_ids,
            min(max_parts, budget),
            coarse_part_graph,
            STITCH_SEED_COUNT,
            score_ranked_ids,
        )
    if STITCH_STRATEGY == "neighbor_rerank":
        return _select_neighbor_rerank_stitch_ids(
            ranked_ids,
            max_parts,
            coarse_part_graph,
            STITCH_SEED_COUNT,
        )
    return _select_ranked_stitch_ids(ranked_ids, max_parts, coarse_part_graph)


def _ensure_graph_features(graph, original_data, device):
    graph_dev = graph.to(device)
    if graph_dev.x is None:
        if hasattr(graph_dev, 'global_id') and graph_dev.global_id is not None:
            gids = graph_dev.global_id.to(original_data.x.device)
            graph_dev.x = original_data.x[gids].to(device)
        else:
            graph_dev.x = torch.zeros(graph_dev.num_nodes, original_data.x.size(1), device=device)
    return graph_dev


def _get_fine_embedding(fid, hierarchy, data, encoder, device):
    cache = hierarchy.setdefault('_fine_embedding_cache', {})
    fid = int(fid)
    if fid in cache:
        return cache[fid]
    graph = hierarchy['fine_graphs'][fid]
    graph_dev = _ensure_graph_features(graph, data, device)
    emb = get_graph_embedding(graph_dev, encoder, device).detach().cpu()
    cache[fid] = emb
    return emb


def _rank_fine_candidates(zq, ranked_coarse_ids, hierarchy, data, encoder, device):
    """Rank fine partitions inside retrieved coarse candidates by query embedding distance."""
    fine_to_coarse = hierarchy['fine_to_coarse_map']
    coarse_set = set(int(cid) for cid in ranked_coarse_ids)
    candidate_fines = [
        int(fid)
        for fid, coarse_id in fine_to_coarse.items()
        if int(coarse_id) in coarse_set
    ]
    candidate_fines = _unique_ordered(candidate_fines)
    if not candidate_fines:
        return []

    zq_cpu = zq.detach().cpu()
    fine_embeds = torch.cat([
        _get_fine_embedding(fid, hierarchy, data, encoder, device)
        for fid in candidate_fines
    ], dim=0)
    distances = torch.linalg.vector_norm(fine_embeds - zq_cpu, dim=1).tolist()
    ranked = [
        fid
        for _, fid in sorted(zip(distances, candidate_fines), key=lambda item: (item[0], item[1]))
    ]
    return ranked


def _mc_dropout_coarse_seed_ids(query_data, encoder, device, faiss_index, faiss_idx_to_coarse_id):
    if MC_DROPOUT_PASSES <= 0:
        return []

    was_training = encoder.training
    encoder.train()
    qdev = query_data.to(device)
    batch_tensor = torch.zeros(qdev.num_nodes, dtype=torch.long, device=device)
    top_k = min(max(1, MC_DROPOUT_TOP_K), faiss_index.ntotal)
    retrieved = []

    with torch.no_grad():
        for _ in range(MC_DROPOUT_PASSES):
            zq_mc, _ = encoder(qdev.x, qdev.edge_index, batch_tensor)
            _, I_mc = faiss_index.search(zq_mc.detach().cpu().numpy(), top_k)
            retrieved.extend(
                faiss_idx_to_coarse_id.get(int(idx), int(idx))
                for idx in I_mc[0]
                if int(idx) >= 0
            )

    if was_training:
        encoder.train()
    else:
        encoder.eval()
    return _unique_ordered(retrieved)


def _select_fine_ranked_ids(ranked_fine_ids, max_parts):
    return _unique_ordered(ranked_fine_ids)[:max_parts]


def _select_fine_boundary_ids(ranked_fine_ids, max_parts, fine_part_graph, seed_count):
    ranked_fine_ids = _unique_ordered(ranked_fine_ids)
    if fine_part_graph is None or max_parts <= 1:
        return ranked_fine_ids[:max_parts]

    rank_pos = {fid: pos for pos, fid in enumerate(ranked_fine_ids)}
    fallback_rank = len(ranked_fine_ids) + 50
    selected = []
    selected_set = set()

    def add(fid):
        fid = int(fid)
        if fid in selected_set:
            return False
        if fid not in rank_pos:
            return False
        selected.append(fid)
        selected_set.add(fid)
        return True

    seed_limit = min(max_parts, max(1, min(seed_count, len(ranked_fine_ids))))
    for fid in ranked_fine_ids[:seed_limit]:
        add(fid)

    frontier_scores = defaultdict(float)
    for seed in selected[:seed_limit]:
        if not fine_part_graph.has_node(seed):
            continue
        seed_rank_bonus = 1.0 / (1.0 + rank_pos.get(seed, fallback_rank))
        for nb in fine_part_graph.neighbors(seed):
            nb = int(nb)
            if nb in selected_set:
                continue
            if nb not in rank_pos:
                continue
            edge_weight = fine_part_graph[seed][nb].get('weight', 1.0)
            nb_rank_bonus = 1.0 / (1.0 + rank_pos.get(nb, fallback_rank))
            frontier_scores[nb] += np.log1p(float(edge_weight)) * (1.0 + seed_rank_bonus) + nb_rank_bonus

    for fid, _ in sorted(frontier_scores.items(), key=lambda kv: (-kv[1], rank_pos.get(kv[0], fallback_rank), kv[0])):
        add(fid)
        if len(selected) >= max_parts:
            return selected

    for fid in ranked_fine_ids:
        add(fid)
        if len(selected) >= max_parts:
            return selected

    return selected


def _is_fine_stitch_strategy():
    return STITCH_STRATEGY in {"fine_ranked", "fine_boundary", "fine_boundary_expand"}


# ═══════════════════════════════════════════════════════════════════════════════
# STITCH-MODE SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def run_stitch_solver(query_data, query_global_ids, true_coarse_indices,
                      data, adj_t, hierarchy, faiss_index, faiss_idx_to_coarse_id,
                      encoder, device, glasgow_bin):
    """
    Stitch-mode: embed query → FAISS top-20 → stitch partitions → Glasgow solve.
    Returns dict with all timing and result fields.
    """
    result = {}
    total_start = time.time()
    result['solver_timed_out'] = False

    # 1. Embed query
    t_embed = time.time()
    query_dev = query_data.to(device)
    zq = get_graph_embedding(query_dev, encoder, device)
    result['query_embedding_time'] = time.time() - t_embed

    # 2. FAISS coarse search
    t_faiss = time.time()
    score_k_requested = FAISS_SCORE_K if FAISS_SCORE_K > 0 else FAISS_TOP_K
    search_k = min(max(FAISS_TOP_K, score_k_requested), faiss_index.ntotal)
    D, I = faiss_index.search(zq.cpu().numpy(), search_k)
    I_translated = [faiss_idx_to_coarse_id.get(int(idx), int(idx)) for idx in I[0] if int(idx) >= 0]
    result['faiss_coarse_search_time'] = time.time() - t_faiss

    predicted_coarse_idx = I_translated[0] if I_translated else -1
    result['predicted_coarse_idx'] = predicted_coarse_idx
    result['correct_coarse_predicted'] = predicted_coarse_idx in true_coarse_indices

    # Recall metrics over coarse partitions touched by the query.
    true_set = set(int(idx) for idx in true_coarse_indices)
    topk_set = set(int(idx) for idx in I_translated[:FAISS_TOP_K])
    top20_set = set(int(idx) for idx in I_translated[:20])
    covered_topk = sorted(true_set & topk_set)
    missed_topk = sorted(true_set - topk_set)
    covered_top20 = sorted(true_set & top20_set)
    score_pool_ids = I_translated[:min(score_k_requested, len(I_translated))]
    score_pool_set = set(int(idx) for idx in score_pool_ids)
    score_pool_missed = sorted(true_set - score_pool_set)
    rank_by_coarse = {int(cid): rank + 1 for rank, cid in enumerate(I_translated)}
    true_coarse_ranks = {cid: rank_by_coarse.get(cid, -1) for cid in sorted(true_set)}
    found_true_ranks = [rank for rank in true_coarse_ranks.values() if rank > 0]
    result['coarse_seed_k'] = FAISS_TOP_K
    result['coarse_score_k'] = score_k_requested
    result['coarse_score_pool_count'] = len(score_pool_ids)
    result['coarse_score_pool_fullcov'] = bool(true_set) and not score_pool_missed
    result['coarse_score_pool_missed'] = str(score_pool_missed)
    result['true_coarse_count'] = len(true_set)
    result['covered_coarse_at_k'] = len(covered_topk)
    result['missed_coarse_at_k'] = str(missed_topk)
    result['retrieval_complete_at_k'] = bool(true_set) and not missed_topk
    result['fullcov_at_k'] = result['retrieval_complete_at_k']
    result['true_coarse_ranks'] = str(true_coarse_ranks)
    result['max_true_coarse_rank'] = max(found_true_ranks) if found_true_ranks else -1
    result['covered_coarse_at_20'] = len(covered_top20)
    result['coarse_recall_at_k'] = len(covered_topk) / len(true_set) if true_set else 0
    result['coarse_recall_at_20'] = len(covered_top20) / len(true_set) if true_set and FAISS_TOP_K >= 20 else -1
    mc_seed_ids = _mc_dropout_coarse_seed_ids(
        query_data, encoder, device, faiss_index, faiss_idx_to_coarse_id
    )
    mc_seed_set = set(mc_seed_ids)
    mc_missed = sorted(true_set - mc_seed_set)
    result['mc_dropout_passes'] = MC_DROPOUT_PASSES
    result['mc_dropout_top_k'] = MC_DROPOUT_TOP_K
    result['mc_dropout_seed_count'] = len(mc_seed_ids)
    result['mc_dropout_seed_fullcov'] = bool(true_set) and not mc_missed
    result['mc_dropout_seed_missed'] = str(mc_missed)
    true_fine_set = determine_true_fine(query_global_ids, hierarchy.get('node_to_fine_map', {}))
    result['true_fine_count'] = len(true_fine_set)

    # 3. Build fine FAISS (within predicted coarse)
    t_ff = time.time()
    fine_to_coarse = hierarchy['fine_to_coarse_map']
    fine_graphs = hierarchy['fine_graphs']
    fine_part_nodes_map = hierarchy['fine_part_nodes_map']
    candidate_fines = [fid for fid, c in fine_to_coarse.items() if c == predicted_coarse_idx]

    predicted_fine_idx = -1
    if candidate_fines:
        fine_embeds = []
        for fid in candidate_fines:
            fg = fine_graphs[fid]
            fg_dev = fg.to(device)
            if fg_dev.x is None and hasattr(fg_dev, 'global_id') and fg_dev.global_id is not None:
                gids = fg_dev.global_id.to(data.x.device)
                fg_dev.x = data.x[gids].to(device)
            elif fg_dev.x is None:
                fg_dev.x = torch.zeros(fg_dev.num_nodes, data.x.size(1), device=device)
            fine_embeds.append(get_graph_embedding(fg_dev, encoder, device))

        fine_embeds_cat = torch.cat(fine_embeds, dim=0)
        fine_index = faiss.IndexFlatL2(GIN_OUTPUT)
        fine_index.add(fine_embeds_cat.cpu().numpy())
        _, I_fine = fine_index.search(zq.cpu().numpy(), min(FAISS_TOP_K, len(candidate_fines)))
        predicted_fine_idx = candidate_fines[I_fine[0][0]]

    result['faiss_fine_search_time'] = time.time() - t_ff
    result['predicted_fine_idx'] = predicted_fine_idx

    # 4. Stitch and solve with iterative expansion
    coarse_part_nodes_map = hierarchy['coarse_part_nodes_map']
    coarse_part_graph = hierarchy['coarse_part_graph']
    fine_part_graph = hierarchy.get('fine_part_graph')

    ranked_ids = _unique_ordered(I_translated[:FAISS_TOP_K] + mc_seed_ids)
    score_ranked_ids = _unique_ordered(I_translated[:min(score_k_requested, len(I_translated))])
    expanded_coarse_ids = []
    if STITCH_STRATEGY in {"coarse_boundary_expand", "fine_boundary_expand"}:
        coarse_budget = BOUNDARY_EXPAND_COARSE_BUDGET if BOUNDARY_EXPAND_COARSE_BUDGET > 0 else FAISS_TOP_K
        expanded_coarse_ids = _select_coarse_boundary_expand_ids(
            ranked_ids,
            coarse_budget,
            coarse_part_graph,
            STITCH_SEED_COUNT,
            score_ranked_ids,
        )
    else:
        expanded_coarse_ids = ranked_ids
    expanded_coarse_set = set(int(cid) for cid in expanded_coarse_ids)
    expanded_missed = sorted(true_set - expanded_coarse_set)
    result['boundary_expand_coarse_budget'] = (
        BOUNDARY_EXPAND_COARSE_BUDGET if BOUNDARY_EXPAND_COARSE_BUDGET > 0 else FAISS_TOP_K
    )
    result['expanded_coarse_count'] = len(expanded_coarse_ids)
    result['expanded_coarse_fullcov'] = bool(true_set) and not expanded_missed
    result['expanded_missed_coarse'] = str(expanded_missed)

    ranked_fine_ids = []
    if _is_fine_stitch_strategy():
        ranked_fine_ids = _rank_fine_candidates(
            zq,
            expanded_coarse_ids if STITCH_STRATEGY == "fine_boundary_expand" else ranked_ids,
            hierarchy,
            data,
            encoder,
            device,
        )
        rank_by_fine = {int(fid): rank + 1 for rank, fid in enumerate(ranked_fine_ids)}
        true_fine_ranks = {fid: rank_by_fine.get(fid, -1) for fid in sorted(true_fine_set)}
        found_true_fine_ranks = [rank for rank in true_fine_ranks.values() if rank > 0]
        result['true_fine_ranks'] = str(true_fine_ranks)
        result['max_true_fine_rank'] = max(found_true_fine_ranks) if found_true_fine_ranks else -1
        result['fine_candidate_pool_count'] = len(ranked_fine_ids)
        fine_pool_set = set(ranked_fine_ids)
        result['fine_pool_fullcov'] = bool(true_fine_set) and not (true_fine_set - fine_pool_set)
    else:
        result['true_fine_ranks'] = '{}'
        result['max_true_fine_rank'] = -1
        result['fine_candidate_pool_count'] = 0
        result['fine_pool_fullcov'] = False

    if _is_fine_stitch_strategy():
        expansion_unit_count = len(ranked_fine_ids)
    elif STITCH_STRATEGY == "coarse_boundary_expand":
        expansion_unit_count = len(expanded_coarse_ids)
    else:
        expansion_unit_count = FAISS_TOP_K
    expansion_levels = _stitch_levels(expansion_unit_count)
    solver_found = False
    solver_time = 0.0

    for max_parts, level_name in expansion_levels:
        # Build stitch
        selected_fine_ids = []
        if _is_fine_stitch_strategy():
            if STITCH_STRATEGY in {"fine_boundary", "fine_boundary_expand"}:
                selected_fine_ids = _select_fine_boundary_ids(
                    ranked_fine_ids,
                    max_parts,
                    fine_part_graph,
                    STITCH_SEED_COUNT,
                )
            else:
                selected_fine_ids = _select_fine_ranked_ids(ranked_fine_ids, max_parts)
            selected_fine_set = set(int(fid) for fid in selected_fine_ids)
            selected_coarse_set = {
                int(fine_to_coarse[fid])
                for fid in selected_fine_ids
                if fid in fine_to_coarse
            }
            candidate_covered = sorted(true_set & selected_coarse_set)
            candidate_missed = sorted(true_set - selected_coarse_set)
            fine_candidate_covered = sorted(true_fine_set & selected_fine_set)
            fine_candidate_missed = sorted(true_fine_set - selected_fine_set)
            candidate_fullcov = bool(true_fine_set) and not fine_candidate_missed
            level_name = f"fine-top-{max_parts}"
            all_nodes = [
                fine_part_nodes_map[fid]
                for fid in selected_fine_ids
                if fid in fine_part_nodes_map
            ]
        else:
            stitched_ids = _select_stitch_ids(ranked_ids, max_parts, coarse_part_graph, score_ranked_ids)
            stitched_set = set(int(idx) for idx in stitched_ids)
            candidate_covered = sorted(true_set & stitched_set)
            candidate_missed = sorted(true_set - stitched_set)
            fine_candidate_covered = []
            fine_candidate_missed = sorted(true_fine_set)
            candidate_fullcov = bool(true_set) and not candidate_missed
            selected_coarse_set = stitched_set
            selected_fine_ids = []
            all_nodes = [
                coarse_part_nodes_map[cidx]
                for cidx in stitched_ids
                if cidx in coarse_part_nodes_map
            ]

        result['candidate_coarse_count'] = len(selected_coarse_set)
        result['candidate_covered_coarse'] = len(candidate_covered)
        result['candidate_missed_coarse'] = str(candidate_missed)
        result['candidate_coarse_fullcov'] = bool(true_set) and not candidate_missed
        result['candidate_fine_count'] = len(selected_fine_ids)
        result['candidate_covered_fine'] = len(fine_candidate_covered)
        result['candidate_missed_fine'] = str(fine_candidate_missed)
        result['candidate_fine_fullcov'] = bool(true_fine_set) and not fine_candidate_missed
        result['candidate_fullcov'] = candidate_fullcov
        result['stitch_strategy'] = STITCH_STRATEGY

        if REQUIRE_CANDIDATE_FULLCOV and not candidate_fullcov:
            result['solver_level'] = f"{level_name}-skipped-incomplete"
            result['stitched_nodes'] = 0
            continue

        if not all_nodes:
            continue

        stitched_nodes = torch.unique(torch.cat(all_nodes))
        result['pre_prune_stitched_nodes'] = len(stitched_nodes)
        if PRUNE_TARGET_BY_QUERY_LABELS:
            stitched_nodes = prune_target_by_query_labels(query_data, stitched_nodes, adj_t, data)
        result['pruned_stitched_nodes'] = len(stitched_nodes)
        stitched_graph = extract_subgraph(adj_t, stitched_nodes, data)
        if stitched_graph is None or stitched_graph.num_nodes == 0:
            continue

        stitched_gids = stitched_graph.global_id if hasattr(stitched_graph, 'global_id') else stitched_nodes

        # Run Glasgow solver
        t_solver = time.time()
        solver_result = glasgow_solve(
            query_data=query_data,
            target_data=stitched_graph,
            query_global_ids=query_global_ids,
            target_global_ids=stitched_gids,
            max_solutions=100,
            timeout_seconds=SOLVER_TIMEOUT,
            binary_path=glasgow_bin,
        )
        solver_time += time.time() - t_solver

        if solver_result.found:
            solver_found = True
            result['perfect_solution_found'] = True
            result['time_to_first_solution'] = solver_result.time_to_first_solution
            result['first_solution_accuracy'] = solver_result.first_solution_accuracy
            result['best_accuracy'] = solver_result.best_accuracy
            result['time_to_best_solution'] = solver_result.latency_seconds
            result['solution_num_for_best_accuracy'] = solver_result.num_solutions
            result['total_solutions_in_timeout'] = solver_result.num_solutions
            result['solver_level'] = level_name
            result['stitched_nodes'] = len(stitched_nodes)
            break
        elif solver_result.timed_out:
            result['solver_level'] = level_name
            result['stitched_nodes'] = len(stitched_nodes)
            result['solver_timed_out'] = True
            break  # Don't expand on timeout

    if not solver_found:
        result['perfect_solution_found'] = False
        result['time_to_first_solution'] = -1.0
        result['first_solution_accuracy'] = -1.0
        result['best_accuracy'] = -1.0
        result['time_to_best_solution'] = -1.0
        result['solution_num_for_best_accuracy'] = 0
        result['total_solutions_in_timeout'] = 0
        result.setdefault('solver_level', 'none')
        result.setdefault('stitched_nodes', 0)

    result['solver_time'] = solver_time
    result['total_time'] = time.time() - total_start
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ORACLE-PARTITION SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def run_oracle_solver(query_data, query_global_ids, true_coarse_indices,
                      data, adj_t, hierarchy, glasgow_bin):
    """
    Oracle mode: stitch exactly the true coarse partitions touched by the query.
    If this fails, the bottleneck is not FAISS retrieval; it is verification,
    query construction, labels, or the size/noise of even the true partition union.
    """
    result = {}
    total_start = time.time()
    true_set = set(int(idx) for idx in true_coarse_indices)

    result['query_embedding_time'] = 0.0
    result['faiss_coarse_search_time'] = 0.0
    result['faiss_fine_search_time'] = 0.0
    result['predicted_coarse_idx'] = -1
    result['predicted_fine_idx'] = -1
    result['correct_coarse_predicted'] = False
    result['true_coarse_count'] = len(true_set)
    result['covered_coarse_at_k'] = len(true_set)
    result['covered_coarse_at_20'] = len(true_set)
    result['missed_coarse_at_k'] = '[]'
    result['retrieval_complete_at_k'] = bool(true_set)
    result['fullcov_at_k'] = result['retrieval_complete_at_k']
    result['coarse_recall_at_k'] = 1.0 if true_set else 0.0
    result['coarse_recall_at_20'] = 1.0 if true_set else 0.0

    coarse_part_nodes_map = hierarchy['coarse_part_nodes_map']
    all_nodes = [
        coarse_part_nodes_map[cidx]
        for cidx in sorted(true_set)
        if cidx in coarse_part_nodes_map
    ]

    if not all_nodes:
        result['perfect_solution_found'] = False
        result['time_to_first_solution'] = -1.0
        result['first_solution_accuracy'] = -1.0
        result['best_accuracy'] = -1.0
        result['time_to_best_solution'] = -1.0
        result['solution_num_for_best_accuracy'] = 0
        result['total_solutions_in_timeout'] = 0
        result['solver_timed_out'] = False
        result['solver_time'] = 0.0
        result['total_time'] = time.time() - total_start
        result['solver_level'] = 'oracle-empty'
        result['stitched_nodes'] = 0
        return result

    stitched_nodes = torch.unique(torch.cat(all_nodes))
    stitched_graph = extract_subgraph(adj_t, stitched_nodes, data)
    if stitched_graph is None or stitched_graph.num_nodes == 0:
        result['perfect_solution_found'] = False
        result['time_to_first_solution'] = -1.0
        result['first_solution_accuracy'] = -1.0
        result['best_accuracy'] = -1.0
        result['time_to_best_solution'] = -1.0
        result['solution_num_for_best_accuracy'] = 0
        result['total_solutions_in_timeout'] = 0
        result['solver_timed_out'] = False
        result['solver_time'] = 0.0
        result['total_time'] = time.time() - total_start
        result['solver_level'] = 'oracle-invalid'
        result['stitched_nodes'] = 0
        return result
    stitched_gids = stitched_graph.global_id if hasattr(stitched_graph, 'global_id') else stitched_nodes

    t_solver = time.time()
    solver_result = glasgow_solve(
        query_data=query_data,
        target_data=stitched_graph,
        query_global_ids=query_global_ids,
        target_global_ids=stitched_gids,
        max_solutions=100,
        timeout_seconds=SOLVER_TIMEOUT,
        binary_path=glasgow_bin,
    )
    solver_time = time.time() - t_solver

    result['perfect_solution_found'] = solver_result.found
    result['time_to_first_solution'] = solver_result.time_to_first_solution if solver_result.found else -1.0
    result['first_solution_accuracy'] = solver_result.first_solution_accuracy
    result['best_accuracy'] = solver_result.best_accuracy
    result['time_to_best_solution'] = solver_result.latency_seconds if solver_result.found else -1.0
    result['solution_num_for_best_accuracy'] = solver_result.num_solutions
    result['total_solutions_in_timeout'] = solver_result.num_solutions
    result['solver_timed_out'] = solver_result.timed_out
    result['solver_time'] = solver_time
    result['total_time'] = time.time() - total_start
    result['solver_level'] = 'oracle-true-parts'
    result['stitched_nodes'] = len(stitched_nodes)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FULL-GRAPH SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_graph_solver(query_data, query_global_ids, data, glasgow_bin, dataset_name):
    """
    Full-graph mode: run Glasgow directly on the entire graph.
    Returns dict with timing and result fields.
    """
    result = {}
    total_start = time.time()

    full_gids = torch.arange(data.num_nodes)

    t_solver = time.time()
    solver_result = glasgow_solve(
        query_data=query_data,
        target_data=data,
        query_global_ids=query_global_ids,
        target_global_ids=full_gids,
        max_solutions=100,
        timeout_seconds=SOLVER_TIMEOUT,
        binary_path=glasgow_bin,
        target_name=f"{dataset_name}_full"
    )
    solver_time = time.time() - t_solver

    result['perfect_solution_found'] = solver_result.found
    result['time_to_first_solution'] = solver_result.time_to_first_solution if solver_result.found else -1.0
    result['first_solution_accuracy'] = solver_result.first_solution_accuracy
    result['best_accuracy'] = solver_result.best_accuracy
    result['time_to_best_solution'] = solver_result.latency_seconds if solver_result.found else -1.0
    result['solution_num_for_best_accuracy'] = solver_result.num_solutions
    result['total_solutions_in_timeout'] = solver_result.num_solutions
    result['solver_timed_out'] = solver_result.timed_out
    result['solver_time'] = solver_time
    result['total_time'] = time.time() - total_start

    # Not applicable for full graph mode
    result['query_embedding_time'] = 0.0
    result['faiss_coarse_search_time'] = 0.0
    result['faiss_fine_search_time'] = 0.0
    result['predicted_coarse_idx'] = -1
    result['correct_coarse_predicted'] = False
    result['predicted_fine_idx'] = -1
    result['true_coarse_count'] = 0
    result['covered_coarse_at_k'] = 0
    result['covered_coarse_at_20'] = 0
    result['missed_coarse_at_k'] = ''
    result['retrieval_complete_at_k'] = False
    result['fullcov_at_k'] = False
    result['coarse_recall_at_k'] = 0.0
    result['coarse_recall_at_20'] = 0.0
    result['solver_level'] = 'full'
    result['stitched_nodes'] = data.num_nodes

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(dataset_name, num_queries=100, output_csv=None, glasgow_bin=None,
                  query_type='all', split_output=False, target_sizes=None,
                  skip_full=False, faiss_top_k=20, solver_timeout=60.0,
                  model_path=None, seed=42, include_oracle=False,
                  stitch_strategy='ranked', stitch_levels=None,
                  stitch_seed_count=20, faiss_score_k=0,
                  boundary_expand_coarse_budget=0,
                  mc_dropout_passes=0, mc_dropout_top_k=20,
                  require_candidate_fullcov=False,
                  prune_target_by_query_labels=False):
    """Run full benchmark for a single dataset."""
    global FAISS_TOP_K, FAISS_SCORE_K, SOLVER_TIMEOUT, STITCH_STRATEGY, STITCH_LEVELS, STITCH_SEED_COUNT, BOUNDARY_EXPAND_COARSE_BUDGET, MC_DROPOUT_PASSES, MC_DROPOUT_TOP_K, REQUIRE_CANDIDATE_FULLCOV, PRUNE_TARGET_BY_QUERY_LABELS
    FAISS_TOP_K = int(faiss_top_k)
    FAISS_SCORE_K = int(faiss_score_k or 0)
    SOLVER_TIMEOUT = float(solver_timeout)
    STITCH_STRATEGY = stitch_strategy
    STITCH_LEVELS = [int(s) for s in stitch_levels.split(',')] if stitch_levels else None
    STITCH_SEED_COUNT = int(stitch_seed_count)
    BOUNDARY_EXPAND_COARSE_BUDGET = int(boundary_expand_coarse_budget or 0)
    MC_DROPOUT_PASSES = int(mc_dropout_passes or 0)
    MC_DROPOUT_TOP_K = int(mc_dropout_top_k or FAISS_TOP_K)
    REQUIRE_CANDIDATE_FULLCOV = bool(require_candidate_fullcov)
    PRUNE_TARGET_BY_QUERY_LABELS = bool(prune_target_by_query_labels)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if glasgow_bin is None:
        glasgow_bin = GLASGOW_BIN
    if target_sizes is None:
        target_sizes = [20]

    cfg = dict(DATASET_CONFIGS[dataset_name])
    if model_path:
        cfg['model'] = model_path
    print(f"\n{'='*70}")
    print(f"  BENCHMARKING: {dataset_name.upper()}")
    print(f"  Queries: {num_queries} | Timeout: {SOLVER_TIMEOUT}s | Solver: Glasgow | FAISS seed K={FAISS_TOP_K}")
    print(f"  Query types: {query_type}")
    print(f"  Target sizes: {target_sizes}")
    print(f"  Model: {cfg['model']}")
    print(f"  Seed: {seed}")
    print(f"  Include oracle: {include_oracle}")
    print(f"  Stitch strategy: {STITCH_STRATEGY}")
    print(f"  Stitch levels: {STITCH_LEVELS or 'default'}")
    print(f"  FAISS score K: {FAISS_SCORE_K or FAISS_TOP_K}")
    print(f"  Boundary expand coarse budget: {BOUNDARY_EXPAND_COARSE_BUDGET or FAISS_TOP_K}")
    print(f"  MC dropout passes/top-k: {MC_DROPOUT_PASSES}/{MC_DROPOUT_TOP_K}")
    print(f"  Require candidate FullCov before solving: {REQUIRE_CANDIDATE_FULLCOV}")
    print(f"  Prune target by query labels: {PRUNE_TARGET_BY_QUERY_LABELS}")
    print(f"{'='*70}\n")

    # ── Load dataset ──
    print("[1/5] Loading dataset...", flush=True)
    t0 = time.time()
    data = load_dataset(cfg['loader'])
    dataset_load_time = time.time() - t0

    # ── Load model ──
    print("[2/5] Loading model...", flush=True)
    encoder, model_load_time = load_model(cfg['model'], data.x.size(1), DEVICE)

    # ── Build hierarchy ──
    print("[3/5] Building hierarchy...", flush=True)
    hierarchy, partition_time = build_hierarchy(data, cfg['coarse'], cfg['fine'])

    # ── Build FAISS index ──
    print("[4/5] Building FAISS index...", flush=True)
    edge_values = data.edge_type.long() if hasattr(data, "edge_type") and data.edge_type is not None else None
    adj_t = SparseTensor.from_edge_index(
        data.edge_index,
        edge_attr=edge_values,
        sparse_sizes=(data.num_nodes, data.num_nodes),
    )
    faiss_index, faiss_idx_to_coarse_id, faiss_build_time = build_faiss_index(
        data, hierarchy, encoder, DEVICE)

    # ── Generate queries ──
    print("[5/5] Generating queries and running solver...", flush=True)

    # Determine which types to generate
    active_types = {'single', 'multi_fine', 'multi_coarse', 'k_hop'} if query_type == 'all' else {query_type}
    n_single = num_queries if 'single' in active_types else 0
    n_mfine = num_queries if 'multi_fine' in active_types else 0
    n_mcoarse = num_queries if 'multi_coarse' in active_types else 0
    n_khop = num_queries if 'k_hop' in active_types else 0

    coarse_ids = sorted(hierarchy['coarse_part_nodes_map'].keys())
    node_to_coarse = hierarchy['node_to_coarse_map']

    all_rows = []

    # CSV header (matching existing format + new fields)
    header = [
        'query_name', 'query_type', 'solver_mode', 'dataset',
        'query_generator_version',
        'anchor_coarse_idx', 'predicted_coarse_idx', 'predicted_fine_idx',
        'correct_coarse_predicted', 'fullcov_at_k', 'coarse_recall_at_k', 'coarse_recall_at_20',
        'true_coarse_count', 'covered_coarse_at_k', 'covered_coarse_at_20',
        'missed_coarse_at_k', 'retrieval_complete_at_k',
        'coarse_seed_k', 'coarse_score_k', 'coarse_score_pool_count',
        'coarse_score_pool_fullcov', 'coarse_score_pool_missed',
        'mc_dropout_passes', 'mc_dropout_top_k', 'mc_dropout_seed_count',
        'mc_dropout_seed_fullcov', 'mc_dropout_seed_missed',
        'boundary_expand_coarse_budget', 'expanded_coarse_count',
        'expanded_coarse_fullcov', 'expanded_missed_coarse',
        'true_coarse_ranks', 'max_true_coarse_rank',
        'true_fine_count', 'true_fine_ranks', 'max_true_fine_rank',
        'fine_candidate_pool_count', 'fine_pool_fullcov',
        'true_coarse_indices', 'query_nodes', 'query_seed',
        'query_embedding_time', 'faiss_coarse_search_time', 'faiss_fine_search_time',
        'perfect_solution_found', 'time_to_first_solution',
        'first_solution_accuracy', 'best_accuracy',
        'time_to_best_solution', 'solution_num_for_best_accuracy',
        'total_solutions_in_timeout', 'solver_time', 'total_time',
        'solver_level', 'stitched_nodes', 'solver_timed_out',
        'stitch_strategy', 'candidate_coarse_count', 'candidate_fullcov',
        'candidate_coarse_fullcov', 'candidate_covered_coarse', 'candidate_missed_coarse',
        'candidate_fine_count', 'candidate_fine_fullcov',
        'candidate_covered_fine', 'candidate_missed_fine',
        'pre_prune_stitched_nodes', 'pruned_stitched_nodes',
        'prune_target_by_query_labels',
        'require_candidate_fullcov',
        'query_size',  # target size used for query generation
        # Infrastructure timing (same for all queries in a dataset)
        'model_load_time', 'partition_time', 'faiss_build_time',
        'faiss_top_k', 'faiss_score_k', 'solver_timeout', 'model_path',
    ]

    query_idx = 0

    def process_query(qtype, query_data, query_global_ids, true_coarse, query_label):
        """Run stitch (and optionally full-graph) modes for a query."""
        nonlocal query_idx
        rows = []

        modes = ['stitch'] if skip_full else ['stitch', 'full']
        if include_oracle:
            modes.insert(1, 'oracle')
        for mode in modes:
            row = {
                'query_name': f"{qtype}_{query_idx}",
                'query_type': qtype,
                'solver_mode': mode,
                'dataset': dataset_name,
                'query_generator_version': QUERY_GENERATOR_VERSION,
                'anchor_coarse_idx': min(true_coarse) if true_coarse else -1,
                'true_coarse_count': len(true_coarse),
                'true_coarse_indices': str(sorted(true_coarse)) if true_coarse else '[]',
                'query_nodes': query_data.num_nodes,
                'query_seed': seed,
                'model_load_time': model_load_time,
                'partition_time': partition_time,
                'faiss_build_time': faiss_build_time,
                'faiss_top_k': FAISS_TOP_K,
                'faiss_score_k': FAISS_SCORE_K or FAISS_TOP_K,
                'boundary_expand_coarse_budget': BOUNDARY_EXPAND_COARSE_BUDGET or FAISS_TOP_K,
                'mc_dropout_passes': MC_DROPOUT_PASSES,
                'mc_dropout_top_k': MC_DROPOUT_TOP_K,
                'solver_timeout': SOLVER_TIMEOUT,
                'model_path': cfg['model'],
                'solver_timed_out': False,
                'require_candidate_fullcov': REQUIRE_CANDIDATE_FULLCOV,
                'prune_target_by_query_labels': PRUNE_TARGET_BY_QUERY_LABELS,
            }

            if mode == 'stitch':
                res = run_stitch_solver(
                    query_data, query_global_ids, true_coarse,
                    data, adj_t, hierarchy, faiss_index, faiss_idx_to_coarse_id,
                    encoder, DEVICE, glasgow_bin
                )
            elif mode == 'oracle':
                res = run_oracle_solver(
                    query_data, query_global_ids, true_coarse,
                    data, adj_t, hierarchy, glasgow_bin
                )
            else:
                res = run_full_graph_solver(
                    query_data, query_global_ids, data, glasgow_bin, dataset_name
                )

            row.update(res)
            rows.append(row)

            found = '✓' if res.get('perfect_solution_found', False) else '✗'
            t = res.get('solver_time', 0)
            print(f"    [{mode:6s}] {found} {t:.1f}s", flush=True)

        query_idx += 1
        return rows

    # ── Single partition queries ──
    if n_single > 0:
        for tsize in target_sizes:
            print(f"\n--- Generating {n_single} SINGLE queries (size={tsize}) ---", flush=True)
            generated = 0
            for _ in range(n_single * 5):  # retry budget
                if generated >= n_single:
                    break
                cid = random.choice(coarse_ids)
                res = generate_single_partition_query(data, adj_t, hierarchy['coarse_part_nodes_map'],
                                                      cid, target_size=tsize)
                if res is None:
                    continue
                query, q_nodes, true_coarse, qtype = res
                print(f"  [{generated+1}/{n_single}] single (P={cid}, Q={query.num_nodes} nodes, sz={tsize})", flush=True)
                rows = process_query('single', query, q_nodes, true_coarse, f"single_{cid}")
                for r in rows: r['query_size'] = tsize
                all_rows.extend(rows)
                generated += 1

    # ── Multi-fine queries ──
    if n_mfine > 0:
        # Pre-build G_nx once for this dataset
        try:
            G_nx = to_networkx(data, to_undirected=True)
        except Exception:
            G_nx = nx.Graph()

        for tsize in target_sizes:
            print(f"\n--- Generating {n_mfine} MULTI_FINE queries (size={tsize}) ---", flush=True)
            generated = 0
            for _ in range(n_mfine * 5):
                if generated >= n_mfine:
                    break
                cid = random.choice(coarse_ids)
                res = generate_multi_fine_query(
                    data, adj_t, G_nx,
                    hierarchy['fine_part_nodes_map'],
                    hierarchy['fine_to_coarse_map'],
                    cid, target_size=tsize
                )
                if res is None:
                    continue
                query, q_nodes, true_coarse, qtype = res
                print(f"  [{generated+1}/{n_mfine}] multi_fine (P={cid}, Q={query.num_nodes} nodes, sz={tsize})", flush=True)
                rows = process_query('multi_fine', query, q_nodes, true_coarse, f"mfine_{generated}")
                for r in rows: r['query_size'] = tsize
                all_rows.extend(rows)
                generated += 1

    # ── Multi-coarse queries ──
    if n_mcoarse > 0:
        for tsize in target_sizes:
            print(f"\n--- Generating {n_mcoarse} MULTI_COARSE queries (size={tsize}) ---", flush=True)
            generated = 0
            for _ in range(n_mcoarse * 5):
                if generated >= n_mcoarse:
                    break
                res = generate_multi_coarse_query(data, adj_t, hierarchy['coarse_part_nodes_map'],
                                                  hierarchy['coarse_part_graph'],
                                                  hierarchy.get('fine_part_nodes_map'),
                                                  hierarchy.get('fine_to_coarse_map'),
                                                  target_size=tsize)
                if res is None:
                    continue
                query, q_nodes, true_coarse, qtype = res
                print(f"  [{generated+1}/{n_mcoarse}] multi_coarse (Q={query.num_nodes} nodes, sz={tsize}, "
                      f"parts={sorted(true_coarse)})", flush=True)
                rows = process_query('multi_coarse', query, q_nodes, true_coarse, f"mcoarse_{generated}")
                for r in rows: r['query_size'] = tsize
                all_rows.extend(rows)
                generated += 1

    # ── K-hop queries ──
    if n_khop > 0:
        for tsize in target_sizes:
            print(f"\n--- Generating {n_khop} K_HOP queries (size={tsize}) ---", flush=True)
            generated = 0
            for _ in range(n_khop * 5):
                if generated >= n_khop:
                    break
                res = generate_k_hop_query(data, adj_t, target_size=tsize)
                if res is None:
                    continue
                query, q_nodes, _, qtype = res
                true_coarse = determine_true_coarse(q_nodes, node_to_coarse)
                if not true_coarse:
                    continue
                print(f"  [{generated+1}/{n_khop}] k_hop (Q={query.num_nodes} nodes, sz={tsize}, "
                      f"parts={len(true_coarse)})", flush=True)
                rows = process_query('k_hop', query, q_nodes, true_coarse, f"khop_{generated}")
                for r in rows: r['query_size'] = tsize
                all_rows.extend(rows)
                generated += 1

    # ── Write CSV ──
    if output_csv is None:
        output_csv = f"glasgow_benchmark_{dataset_name}.csv"

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n{'='*70}")
    print(f"  RESULTS SAVED: {output_csv}")
    mode_count = 1 if skip_full else 2
    if include_oracle:
        mode_count += 1
    unique_count = len(all_rows) // mode_count if mode_count else len(all_rows)
    print(f"  Total rows: {len(all_rows)} ({unique_count} unique queries x {mode_count} mode(s))")
    print(f"{'='*70}")

    # ── Split output into subgnn/vanilla CSVs (matching bench_data format) ──
    if split_output:
        out_dir = os.path.dirname(output_csv) or '.'
        prefix = query_type + '_' if query_type != 'all' else ''

        stitch_rows = [r for r in all_rows if r.get('solver_mode') == 'stitch']
        oracle_rows = [r for r in all_rows if r.get('solver_mode') == 'oracle']
        full_rows = [r for r in all_rows if r.get('solver_mode') == 'full']

        for fname, rows in [('subgnn_benchmark_results.csv', stitch_rows),
                            ('oracle_benchmark_results.csv', oracle_rows),
                            ('vanilla_benchmark_results.csv', full_rows)]:
            out_path = os.path.join(out_dir, f"{prefix}{fname}")
            with open(out_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"  Split output: {out_path} ({len(rows)} rows)")

    # ── Print summary ──
    print(f"\n  Infrastructure Timing:")
    print(f"    Model Load:      {model_load_time:.2f}s")
    print(f"    Partitioning:    {partition_time:.2f}s")
    print(f"    FAISS Build:     {faiss_build_time:.2f}s")

    for mode in ['stitch', 'oracle', 'full']:
        mode_rows = [r for r in all_rows if r['solver_mode'] == mode]
        if not mode_rows:
            continue
        found = sum(1 for r in mode_rows if r.get('perfect_solution_found', False))
        avg_time = np.mean([r.get('solver_time', 0) for r in mode_rows])
        timed_out = sum(1 for r in mode_rows if r.get('solver_timed_out', False))
        print(f"\n  Glasgow on {mode.upper()}:")
        print(f"    Found:     {found}/{len(mode_rows)} ({100*found/len(mode_rows):.1f}%)")
        print(f"    Avg Time:  {avg_time:.2f}s")
        print(f"    Timeouts:  {timed_out}/{len(mode_rows)}")

    return all_rows


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Glasgow Solver Benchmark")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'corafull', 'arxiv', 'mag', 'pubmed', 'citeseer', 'physics'],
                        help='Dataset to benchmark (default: all)')
    parser.add_argument('--queries', type=int, default=100,
                        help='Number of queries per type (default: 100)')
    parser.add_argument('--query-type', type=str, default='all',
                        choices=['all', 'single', 'multi_fine', 'multi_coarse', 'k_hop'],
                        help='Query type to benchmark (default: all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: glasgow_benchmark_<dataset>.csv)')
    parser.add_argument('--split-output', action='store_true',
                        help='Split output into subgnn/vanilla CSVs')
    parser.add_argument('--target-sizes', type=str, default='20,50,100',
                        help='Comma-separated query node sizes (default: 20,50,100)')
    parser.add_argument('--skip-full', action='store_true',
                        help='Skip full-graph solver (for large datasets like arxiv)')
    parser.add_argument('--glasgow_bin', type=str, default=None,
                        help='Path to glasgow_subgraph_solver binary')
    parser.add_argument('--faiss-top-k', type=int, default=20,
                        help='Number of coarse partitions used as the retrieval seed set (default: 20)')
    parser.add_argument('--faiss-score-k', type=int, default=0,
                        help='Optional deeper FAISS pool used only as ranking/tie-break scores for expansion')
    parser.add_argument('--solver-timeout', type=float, default=60.0,
                        help='Glasgow timeout per solver call in seconds (default: 60)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Override model checkpoint path for the selected dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible query generation (default: 42)')
    parser.add_argument('--include-oracle', action='store_true',
                        help='Also run Glasgow on exactly the true coarse partitions')
    parser.add_argument('--stitch-strategy', type=str, default='ranked',
                        choices=['ranked', 'neighbor_rerank', 'coarse_boundary_expand',
                                 'fine_ranked', 'fine_boundary', 'fine_boundary_expand'],
                        help='Candidate selection strategy for stitch mode')
    parser.add_argument('--stitch-levels', type=str, default=None,
                        help='Comma-separated candidate coarse-partition counts to try')
    parser.add_argument('--stitch-seed-count', type=int, default=20,
                        help='Number of top ranked coarse partitions kept as seeds for neighbor/boundary expansion')
    parser.add_argument('--boundary-expand-coarse-budget', type=int, default=0,
                        help='Final coarse candidate budget for boundary expansion; 0 uses --faiss-top-k')
    parser.add_argument('--mc-dropout-passes', type=int, default=0,
                        help='Optional stochastic query embedding passes whose top-k coarse hits are unioned into the seed set')
    parser.add_argument('--mc-dropout-top-k', type=int, default=20,
                        help='Top-k coarse partitions retrieved per MC dropout pass')
    parser.add_argument('--require-candidate-fullcov', action='store_true',
                        help='Benchmark-only guard: run Glasgow only after the stitched candidate covers all true coarse partitions')
    parser.add_argument('--prune-target-by-query-labels', action='store_true',
                        help='Prune stitched target nodes to labels present in the query before Glasgow')
    args = parser.parse_args()

    target_sizes = [int(s) for s in args.target_sizes.split(',')]

    if args.dataset == 'all':
        datasets = ['citeseer', 'pubmed', 'corafull', 'physics', 'arxiv']  # smallest first
    else:
        datasets = [args.dataset]

    all_results = []
    for ds in datasets:
        output = args.output or f"glasgow_benchmark_{ds}.csv"
        results = run_benchmark(ds, num_queries=args.queries, output_csv=output,
                                glasgow_bin=args.glasgow_bin,
                                query_type=args.query_type,
                                split_output=args.split_output,
                                target_sizes=target_sizes,
                                skip_full=args.skip_full,
                                faiss_top_k=args.faiss_top_k,
                                faiss_score_k=args.faiss_score_k,
                                solver_timeout=args.solver_timeout,
                                model_path=args.model_path,
                                seed=args.seed,
                                include_oracle=args.include_oracle,
                                stitch_strategy=args.stitch_strategy,
                                stitch_levels=args.stitch_levels,
                                stitch_seed_count=args.stitch_seed_count,
                                boundary_expand_coarse_budget=args.boundary_expand_coarse_budget,
                                mc_dropout_passes=args.mc_dropout_passes,
                                mc_dropout_top_k=args.mc_dropout_top_k,
                                require_candidate_fullcov=args.require_candidate_fullcov,
                                prune_target_by_query_labels=args.prune_target_by_query_labels)
        all_results.extend(results)

    # If running all datasets, also write a combined CSV
    if args.dataset == 'all' and all_results:
        combined_csv = "glasgow_benchmark_all.csv"
        header = list(all_results[0].keys())
        with open(combined_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
        print(f"\n  Combined results saved to: {combined_csv}")


if __name__ == '__main__':
    main()
