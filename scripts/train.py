import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="outdated")
import math
import torch
import torch.nn as nn
import random
import time
import queue
import threading
import concurrent.futures
import itertools
import gc
from collections import Counter, defaultdict, deque
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data, HeteroData
from torch_sparse import SparseTensor

# Add parent dir to path to find 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.model import ImprovedSubgraphEncoder, NodeFeatureAugmentor, hierarchical_info_nce_loss
from src.data import make_partitions, convert_hetero_to_homo, make_undirected_fast, build_multiple_hierarchies
from src.sampling import generate_hierarchical_sample
from torch_geometric.datasets import CoraFull
from ogb.nodeproppred import PygNodePropPredDataset

class JigsawDataset(Dataset):
    def __init__(self, original_data, adj_t, hierarchies, batch_size, steps_per_epoch):
        self.original_data = original_data
        self.adj_t = adj_t # Full GPU SparseTensor or CPU if mapped
        self.hierarchies = hierarchies
        
        # Optimize: Pre-convert node_to_coarse_map to GPU tensor for each hierarchy
        self.node_to_coarse_tensors = []
        for h_data in hierarchies:
            node_map_dict = h_data['node_to_coarse_map']
            # Create a tensor initialized with -1 or a valid default
            mapper = torch.full((original_data.num_nodes,), -1, dtype=torch.long, device=original_data.x.device)
            
            keys = torch.tensor(list(node_map_dict.keys()), dtype=torch.long)
            values = torch.tensor(list(node_map_dict.values()), dtype=torch.long)
            # Move to device for assignment
            keys = keys.to(original_data.x.device)
            values = values.to(original_data.x.device)
            mapper[keys] = values
            self.node_to_coarse_tensors.append(mapper)
            
            # Pre-compute reverse map and edges for multi-coarse sampling optimization
            c2f = defaultdict(list)
            f2c = h_data['fine_to_coarse_map']
            for f, c in f2c.items():
                c2f[c].append(f)
            h_data['precomputed_coarse_to_fine'] = c2f
            
            # Pre-compute coarse edges list
            h_data['precomputed_coarse_edges'] = list(h_data['coarse_part_graph'].edges())
            
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
    
    def __len__(self):
        return self.steps_per_epoch

    def generate_sample(self):
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
                    h_data['precomputed_coarse_to_fine'],
                    list(h_data['precomputed_coarse_edges']), # Pass a copy or list to shuffle inside
                    h_data['fine_part_nodes_map'], 
                    h_data['coarse_part_nodes_map'],
                    h_data['coarse_part_graph'],
                    coarse_edge_to_fine_bridges=h_data.get('coarse_edge_to_fine_bridges')
                )
                if sample:
                    # Inject h_idx into metadata
                    sample[3]['hierarchy_idx'] = h_idx
                    return sample
            except RuntimeError:
                continue # Retry on error

    def __getitem__(self, idx):
        return self.generate_sample()

def jigsaw_collate_fn(batch_list):
    gqs = []
    gpos = []
    gcs = []
    metadatas = []
    
    for b in batch_list:
        if b is None: continue 
        # tuple unpacking: (Gq, Gpos, G_coarse_pos, metadata)
        if len(b) >= 4:
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
        else:
            continue

    return Batch.from_data_list(gqs), Batch.from_data_list(gpos), Batch.from_data_list(gcs), metadatas

def load_data(name):
    print(f"[INFO] Loading dataset: {name}")
    if name == "cora":
        dataset = CoraFull(root=DATA_ROOT_CORA)
        data = dataset[0]
        # Standardize attributes
        if not hasattr(data, 'node_types'):
            data.node_types = ['paper'] # Generic type for homogeneous
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
        
    elif name == "arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=DATA_ROOT_ARXIV)
        data = dataset[0]
        if not hasattr(data, 'node_types'):
            data.node_types = ['paper']
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
            
    elif name == "mag":
        dataset = PygNodePropPredDataset(name="ogbn-mag", root=DATA_ROOT_MAG)
        data = convert_hetero_to_homo(dataset[0])
    else: raise ValueError
    
    # Initialize global_id attribute if missing
    if not hasattr(data, 'global_id'):
        data.global_id = torch.arange(data.num_nodes)
        
    print("\n[INFO] Symmetrizing graph...", flush=True)
    data.edge_index = make_undirected_fast(data.edge_index, data.num_nodes)
    
    return data

def init_worker(worker_id):
    torch.set_num_threads(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50) # Steps per epoch
    parser.add_argument("--hierarchies", type=int, default=3)
    parser.add_argument("--fallback", type=int, default=1) # 0=Wait, 1=CPU, 2=GPU
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    device = DEVICE
    data = load_data(args.dataset)
    
    print(f"[INFO] Graph loaded. Keeping structure on CPU to optimize memory usage.", flush=True)
    
    TYPE_DIM = 16; NODE_DIM = 16
    
    # Conditional Augmentor as per user instruction
    if args.dataset == 'mag':
        print("[INFO] Initializing NodeFeatureAugmentor for MAG...", flush=True)
        augmentor = NodeFeatureAugmentor(num_nodes=data.num_nodes, num_types=len(data.node_types), type_dim=TYPE_DIM, node_dim=NODE_DIM).to(device)
        base_feat_dim = data.x.size(1)
        augmented_feat_dim = base_feat_dim + augmentor.added_dim
    else:
        print(f"[INFO] Skipping NodeFeatureAugmentor for {args.dataset}...", flush=True)
        # Dummy augmentor that returns input x
        augmentor = nn.Sequential() 
        base_feat_dim = data.x.size(1)
        augmented_feat_dim = base_feat_dim

    print(f"\n[INFO] Base features: {base_feat_dim}, Final Model Input Dim: {augmented_feat_dim}", flush=True)

    print("[SETUP] Building SparseTensor adjacency for efficient slicing (on CPU)...", flush=True)
    adj_t = SparseTensor(
        row=data.edge_index[0], 
        col=data.edge_index[1], 
        sparse_sizes=(data.num_nodes, data.num_nodes)
    )
    adj_t.csr() 
    print("  - SparseTensor built and on CPU.", flush=True)

    encoder = ImprovedSubgraphEncoder(augmented_feat_dim, 256, 128, dropout=0.1, use_residual=True).to(device)
    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), augmentor.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    
    # Generate Cache Path
    CACHE_PATH = f"v1/data/{args.dataset}_hierarchies.pt"
    
    # Hierarchy Logic
    # Use User-Defined Partition Configs
    cfg = PARTITION_CONFIGS.get(args.dataset, PARTITION_CONFIGS['default'])
    target_coarse = cfg['coarse']
    target_fine = cfg['fine']
    
    print(f"[CONFIG] Target Partitioning for {args.dataset}: Coarse={target_coarse}, Fine={target_fine}")
    
    if os.path.exists(CACHE_PATH):
        print(f"[CACHE] Found cached hierarchies at {CACHE_PATH}. Loading...", flush=True)
        try:
             hierarchies = torch.load(CACHE_PATH)
        except:
             print(f"[CACHE] Failed to load {CACHE_PATH}. Rebuilding...", flush=True)
             hierarchies = build_multiple_hierarchies(data, args.hierarchies, target_coarse, target_fine)
             torch.save(hierarchies, CACHE_PATH)
    else:
        hierarchies = build_multiple_hierarchies(data, args.hierarchies, target_coarse, target_fine)
        torch.save(hierarchies, CACHE_PATH)

    # Move to CPU for parallel worker processes
    print("[INFO] Creating Graph and SparseTensor copies on CPU for parallel sampling...", flush=True)
    data_cpu = data.cpu()
    adj_t_cpu = adj_t.cpu()
    
    print("[INFO] Creating independent CPU hierarchy copy...", flush=True)
    hierarchies_cpu = []
    
    for h_gpu in hierarchies:
        h_cpu = {}
        for k, v in h_gpu.items():
            if isinstance(v, torch.Tensor):
                h_cpu[k] = v.cpu()
            elif isinstance(v, list):
                if len(v) > 0:
                     if isinstance(v[0], Data):
                         new_list = []
                         for item in v:
                             if item is None: new_list.append(None); continue
                             item_cpu = item.cpu() 
                             if hasattr(item, 'adj_t') and item.adj_t is not None:
                                 try: item_cpu.adj_t = item.adj_t.cpu()
                                 except: pass
                             new_list.append(item_cpu)
                         h_cpu[k] = new_list
                     elif hasattr(v[0], 'cpu'):
                         h_cpu[k] = [item.cpu() for item in v]
                     else: h_cpu[k] = v 
                else: h_cpu[k] = []
            elif isinstance(v, dict):
                 new_dict = {}
                 for subk, subv in v.items():
                     if isinstance(subv, torch.Tensor): new_dict[subk] = subv.cpu()
                     else: new_dict[subk] = subv
                 h_cpu[k] = new_dict
            else: h_cpu[k] = v 
        hierarchies_cpu.append(h_cpu)
    
    dataset_cpu = JigsawDataset(data_cpu, adj_t_cpu, hierarchies_cpu, args.batch_size, args.steps)
    
    dataset_fallback = None
    if args.fallback == 1:
        print("[INFO] Fallback Mode 1: CPU.", flush=True)
        dataset_fallback = JigsawDataset(data_cpu, adj_t_cpu, hierarchies_cpu, args.batch_size, args.steps)
    elif args.fallback == 2:
        print("[INFO] Fallback Mode 2: GPU.", flush=True)
        # Note: 'data' is still partly on CPU due to optimization above. 
        # For GPU fallback, we ideally want data on GPU.
        # But for large graphs, we keep features on CPU. 
        # The sampler handles this mixed mode.
        dataset_fallback = JigsawDataset(data, adj_t, hierarchies, args.batch_size, args.steps)

    encoder.train(); augmentor.train()
    
    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        total_loss = 0
        
        batch_loader = DataLoader(
            dataset_cpu, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.workers,
            collate_fn=jigsaw_collate_fn,
            persistent_workers=False,
            prefetch_factor=2,
            worker_init_fn=init_worker
        )
        
        iterator = iter(batch_loader)
        pbar = tqdm(range(args.steps), desc=f"Epoch {epoch+1}/{args.epochs}", unit="step", mininterval=10.0) 
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        def fetch_next():
            try: return next(iterator)
            except StopIteration: return None
        
        current_future = executor.submit(fetch_next)
        
        for step in pbar:
            t_wait_start = time.time()
            try:
                res = current_future.result(timeout=0.1) 
                if res is None: 
                     iterator = iter(batch_loader); current_future = executor.submit(fetch_next); continue
                
                batch_data = res[0], res[1], res[2]
                batch_metadata = res[3]
                current_future = executor.submit(fetch_next)
                
            except concurrent.futures.TimeoutError:
                if args.fallback == 0 or dataset_fallback is None:
                    try:
                        res = current_future.result()
                        if res is None: 
                             iterator = iter(batch_loader); current_future = executor.submit(fetch_next); continue
                        batch_data = (res[0], res[1], res[2])
                        batch_metadata = res[3]
                        current_future = executor.submit(fetch_next)
                    except Exception as e:
                         print(f"Worker Error: {e}"); break
                else:
                    # Fallback generation
                    fallback_samples = []
                    batch_metadata = []
                    for _ in range(args.batch_size):
                        s = dataset_fallback.generate_sample()
                        if s: 
                             cleaned_parts = []
                             for item in s[:3]:
                                 if hasattr(item, 'part_id'): delattr(item, 'part_id')
                                 cleaned_parts.append(item)
                             fallback_samples.append(tuple(cleaned_parts))
                             batch_metadata.append(s[3])
                    
                    if not fallback_samples: continue
                    batch_data = (Batch.from_data_list([s[0] for s in fallback_samples]), 
                                  Batch.from_data_list([s[1] for s in fallback_samples]), 
                                  Batch.from_data_list([s[2] for s in fallback_samples]))
                    for m in batch_metadata: m['source'] = 'fallback'

            query_batch, pos_batch, coarse_pos_batch = batch_data
            query_batch = query_batch.to(device)
            pos_batch = pos_batch.to(device)
            coarse_pos_batch = coarse_pos_batch.to(device)
            
            optimizer.zero_grad()
            try:
                xq = augmentor(query_batch); xp = augmentor(pos_batch); xc = augmentor(coarse_pos_batch)
                zq = encoder(xq, query_batch.edge_index, query_batch.batch)
                z_pos = encoder(xp, pos_batch.edge_index, pos_batch.batch)
                z_coarse = encoder(xc, coarse_pos_batch.edge_index, coarse_pos_batch.batch)
                
                loss = hierarchical_info_nce_loss(zq, z_pos, z_coarse)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item(); pbar.set_postfix({"loss": loss.item()})
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache(); continue
                else: raise e
            
            del query_batch, pos_batch, coarse_pos_batch, xq, xp, xc, zq, z_pos, z_coarse, loss, batch_data
            if step % 20 == 0: gc.collect(); torch.cuda.empty_cache()
            
        avg_loss = total_loss / args.steps
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.6f}")
        
    # Save
    save_path = f"v1/data/{args.dataset}-model-jigsaw.pth"
    torch.save(encoder.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
