"""
Jigsaw GNN Evaluation Framework

Stratified partition evaluation with multiple query types and comprehensive metrics.
Uses VF2 (NetworkX) for subgraph isomorphism verification.
"""

import os
import time
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_sparse import SparseTensor

from src.config import (
    DEVICE, FAISS_TOP_K, 
    GIN_HIDDEN_NEURONS, GIN_OUTPUT_NEURONS,
    QUERY_NODES_MIN, QUERY_NODES_MAX,
    PARTITION_CONFIGS
)
from src.model import ImprovedSubgraphEncoder, get_graph_embedding
from src.vf2_matcher import vf2_verify_subgraph, MatchResult
from src.query_generator import (
    generate_k_hop_query,
    generate_single_partition_query,
    generate_multi_fine_partition_query,
    generate_multi_coarse_partition_query,
)


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Retrieval metrics
    coarse_recall_at_1: float = 0.0
    coarse_recall_at_k: float = 0.0
    fine_recall_at_1: float = 0.0
    fine_recall_at_k: float = 0.0
    
    # Matching metrics
    match_rate: float = 0.0  # % of queries where VF2 found isomorphism
    node_accuracy: float = 0.0  # Average node mapping accuracy
    
    # Precision/Recall/F1 for node matching
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Latency metrics (seconds)
    avg_embedding_latency: float = 0.0
    avg_faiss_latency: float = 0.0
    avg_vf2_latency: float = 0.0
    avg_total_latency: float = 0.0
    
    # Counts
    total_queries: int = 0
    successful_queries: int = 0


def compute_precision_recall_f1(
    predicted_nodes: set,
    ground_truth_nodes: set
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 for node set prediction.
    
    Args:
        predicted_nodes: Set of predicted node IDs
        ground_truth_nodes: Set of ground truth node IDs
        
    Returns:
        (precision, recall, f1)
    """
    if len(predicted_nodes) == 0 and len(ground_truth_nodes) == 0:
        return 1.0, 1.0, 1.0
    if len(predicted_nodes) == 0:
        return 0.0, 0.0, 0.0
    if len(ground_truth_nodes) == 0:
        return 0.0, 0.0, 0.0
    
    intersection = predicted_nodes & ground_truth_nodes
    
    precision = len(intersection) / len(predicted_nodes)
    recall = len(intersection) / len(ground_truth_nodes)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def aggregate_metrics(results: List[Dict]) -> EvaluationMetrics:
    """
    Aggregate individual query results into summary metrics.
    
    Args:
        results: List of result dicts from search_and_verify
        
    Returns:
        EvaluationMetrics summary
    """
    metrics = EvaluationMetrics()
    
    if not results:
        return metrics
    
    successful = [r for r in results if r.get('success', False)]
    metrics.total_queries = len(results)
    metrics.successful_queries = len(successful)
    
    if not successful:
        return metrics
    
    # Retrieval metrics
    metrics.coarse_recall_at_1 = np.mean([r.get('coarse_correct', False) for r in successful])
    metrics.coarse_recall_at_k = np.mean([r.get('coarse_in_top_k', False) for r in successful])
    metrics.fine_recall_at_1 = np.mean([r.get('fine_correct', False) for r in successful])
    metrics.fine_recall_at_k = np.mean([r.get('fine_in_top_k', False) for r in successful])
    
    # Matching metrics
    matched = [r for r in successful if r.get('vf2_found', False)]
    metrics.match_rate = len(matched) / len(successful) if successful else 0
    
    if matched:
        metrics.node_accuracy = np.mean([r.get('node_accuracy', 0) for r in matched])
    
    # Precision/Recall/F1 (from node prediction)
    precisions = [r.get('precision', 0) for r in successful if 'precision' in r]
    recalls = [r.get('recall', 0) for r in successful if 'recall' in r]
    f1s = [r.get('f1', 0) for r in successful if 'f1' in r]
    
    if precisions:
        metrics.precision = np.mean(precisions)
    if recalls:
        metrics.recall = np.mean(recalls)
    if f1s:
        metrics.f1_score = np.mean(f1s)
    
    # Latency metrics
    metrics.avg_embedding_latency = np.mean([r.get('embed_time', 0) for r in successful])
    metrics.avg_faiss_latency = np.mean([r.get('faiss_time', 0) for r in successful])
    metrics.avg_vf2_latency = np.mean([r.get('vf2_time', 0) for r in successful])
    metrics.avg_total_latency = np.mean([r.get('total_time', 0) for r in successful])
    
    return metrics


# =============================================================================
# CHECKPOINTING (Resumable Evaluation)
# =============================================================================

def save_checkpoint(results: list, checkpoint_path: str, completed_keys: set):
    """
    Save evaluation progress for resumability.
    
    Args:
        results: List of result dicts
        checkpoint_path: Path to checkpoint JSON
        completed_keys: Set of completed (query_type, partition_idx) tuples
    """
    import json
    checkpoint_data = {
        'completed_keys': [list(k) for k in completed_keys],  # Convert tuples for JSON
        'results': results
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_path: str):
    """
    Load evaluation progress.
    
    Returns:
        (completed_keys: set of (query_type, partition_idx), results: list)
    """
    import json
    if not os.path.exists(checkpoint_path):
        return set(), []
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    completed_keys = {tuple(k) for k in data.get('completed_keys', [])}
    results = data.get('results', [])
    print(f"[INFO] Loaded checkpoint: {len(completed_keys)} partitions completed, {len(results)} results")
    return completed_keys, results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _extract_subgraph(adj_t: SparseTensor, node_indices: torch.Tensor, original_data: Data) -> Optional[Data]:
    """Helper to extract subgraph using SparseTensor slicing."""
    if node_indices.numel() == 0:
        return None
    
    adj_device = adj_t.device()
    nodes = node_indices.to(adj_device) if node_indices.device != adj_device else node_indices
    
    if nodes.max().item() >= adj_t.sparse_sizes()[0]:
        return None
    
    try:
        sub_adj = adj_t[nodes, nodes]
        row, col, _ = sub_adj.coo()
        edge_index = torch.stack([row, col], dim=0)
    except Exception:
        return None
    
    feat_device = original_data.x.device
    nodes_feat = node_indices.to(feat_device) if node_indices.device != feat_device else node_indices
    x = original_data.x[nodes_feat]
    
    data = Data(x=x.cpu(), edge_index=edge_index.cpu(), num_nodes=len(node_indices))
    data.global_id = node_indices.cpu()
    
    return data


# =============================================================================
# SEARCH AND VERIFICATION
# =============================================================================

def search_and_verify(
    query_data: Data,
    query_global_ids: torch.Tensor,
    true_coarse_indices: set,
    context: Dict[str, Any],
    encoder: ImprovedSubgraphEncoder,
    device: torch.device,
    top_k: int = 5,
    vf2_timeout: float = 30.0,
    ground_truth_target: Optional[Data] = None,  # Stitched ground truth target
    run_baseline: bool = False  # Whether to run VF2 on full graph (SLOW)
) -> Dict[str, Any]:
    """
    Full search pipeline: embed -> FAISS search -> VF2 verify.
    
    VF2 Comparisons:
    1. vf2_predicted: Query vs predicted coarse partition (our method)
    2. vf2_ground_truth: Query vs stitched ground truth target
    3. vf2_baseline: Query vs full graph (optional, very slow)
    
    Returns:
        Dict with all timing and accuracy metrics
    """
    result = {'success': False}
    total_start = time.time()
    
    try:
        # 1. Embed query (apply augmentor for MAG)
        t_embed = time.time()
        query_data = query_data.to(device)
        augmentor = context.get('augmentor')
        original_data = context.get('original_data')
        
        # Fetch features if missing
        if query_data.x is None and original_data is not None:
            if hasattr(query_data, 'global_id') and query_data.global_id is not None:
                global_ids = query_data.global_id.to(original_data.x.device)
                query_data.x = original_data.x[global_ids].to(device)
        
        if augmentor is not None:
            query_data = query_data.clone()
            query_data.x = augmentor(query_data)
        zq = get_graph_embedding(query_data, encoder, device)
        result['embed_time'] = time.time() - t_embed
        
        # 2. Coarse FAISS search
        t_faiss = time.time()
        faiss_coarse = context['faiss_coarse']
        D_coarse, I_coarse = faiss_coarse.search(zq.cpu().numpy(), top_k)
        
        predicted_coarse_idx = I_coarse[0][0]
        result['predicted_coarse'] = int(predicted_coarse_idx)
        result['coarse_correct'] = predicted_coarse_idx in true_coarse_indices
        result['coarse_in_top_k'] = any(idx in true_coarse_indices for idx in I_coarse[0])
        result['faiss_time'] = time.time() - t_faiss
        
        # 3. Fine FAISS search within predicted coarse
        fine_to_coarse = context['fine_to_coarse_map']
        fine_graphs = context['fine_graphs']
        
        candidate_fines = [i for i, c in fine_to_coarse.items() if c == predicted_coarse_idx]
        
        if candidate_fines:
            # NOTE: Fine graphs may have x=None, fetch from original_data
            def embed_fine(g):
                if g is None or g.num_nodes == 0:
                    return torch.zeros(1, 128, device=device)
                g = g.to(device)
                if g.x is None and original_data is not None:
                    if hasattr(g, 'global_id') and g.global_id is not None:
                        global_ids = g.global_id.to(original_data.x.device)
                        g.x = original_data.x[global_ids].to(device)
                    else:
                        g.x = torch.zeros(g.num_nodes, original_data.x.size(1), device=device)
                if augmentor is not None:
                    g = g.clone()
                    g.x = augmentor(g)
                return get_graph_embedding(g, encoder, device)
            
            candidate_embeds = torch.cat([embed_fine(fine_graphs[i]) for i in candidate_fines], dim=0)
            
            faiss_fine = faiss.IndexFlatL2(candidate_embeds.shape[1])
            faiss_fine.add(candidate_embeds.cpu().numpy())
            _, I_fine = faiss_fine.search(zq.cpu().numpy(), min(top_k, len(candidate_fines)))
            
            predicted_fine_idx = candidate_fines[I_fine[0][0]]
            result['predicted_fine'] = predicted_fine_idx
            
            true_fine_indices = context.get('true_fine_indices', set())
            result['fine_correct'] = predicted_fine_idx in true_fine_indices
            result['fine_in_top_k'] = any(candidate_fines[i] in true_fine_indices for i in I_fine[0])
        
        # ================================================================
        # 4. VF2 VERIFICATION - 2 comparisons:
        #    (a) Our Method: Top-3 + neighbors stitching (V1 style)
        #    (b) Baseline: Full graph search
        # ================================================================
        t_vf2 = time.time()
        
        coarse_graphs = context['coarse_graphs']
        coarse_part_nodes_map = context['coarse_part_nodes_map']
        coarse_part_graph = context.get('coarse_part_graph')  # NetworkX graph of coarse partition connectivity
        adj_t = context['adj_t']
        
        # 4a. V1-STYLE STITCHING: Top-3 + their graph neighbors
        # Step 1: Get top-3 from FAISS
        top_k_for_neighbors = 3
        max_stitched_partitions = 20
        
        top_coarse_indices = set(int(idx) for idx in I_coarse[0][:top_k_for_neighbors])
        
        # Step 2: Expand with graph neighbors of top-3
        if coarse_part_graph is not None:
            expanded_indices = set(top_coarse_indices)
            for cidx in top_coarse_indices:
                if coarse_part_graph.has_node(cidx):
                    neighbors = list(coarse_part_graph.neighbors(cidx))
                    expanded_indices.update(neighbors)
            
            # Limit to max 20 partitions (prioritize top FAISS results)
            # Re-rank expanded by their FAISS rank (if in top-K) or add at end
            faiss_ranked = list(I_coarse[0][:top_k])  # Top-20 from FAISS
            stitched_coarse_indices = []
            
            # First add those in both expanded AND top FAISS
            for idx in faiss_ranked:
                if int(idx) in expanded_indices:
                    stitched_coarse_indices.append(int(idx))
                    if len(stitched_coarse_indices) >= max_stitched_partitions:
                        break
            
            # Then add remaining expanded that weren't in top FAISS
            for idx in expanded_indices:
                if idx not in stitched_coarse_indices:
                    stitched_coarse_indices.append(idx)
                    if len(stitched_coarse_indices) >= max_stitched_partitions:
                        break
        else:
            # Fallback: just use top-K from FAISS
            stitched_coarse_indices = [int(idx) for idx in I_coarse[0][:max_stitched_partitions]]
        
        result['stitched_partitions'] = stitched_coarse_indices
        result['num_stitched'] = len(stitched_coarse_indices)
        
        # KEY METRIC: Are ground truth partitions covered by stitched?
        gt_partitions_in_stitched = sum(1 for gt in true_coarse_indices if gt in stitched_coarse_indices)
        result['gt_partitions_covered'] = gt_partitions_in_stitched
        result['gt_partitions_total'] = len(true_coarse_indices)
        result['gt_partition_recall'] = gt_partitions_in_stitched / len(true_coarse_indices) if true_coarse_indices else 0
        result['all_gt_partitions_found'] = (gt_partitions_in_stitched == len(true_coarse_indices))
        
        # Query stats
        result['query_nodes'] = len(query_global_ids) if query_global_ids is not None else 0
        
        # Step 3: Collect all nodes from stitched partitions
        stitched_nodes_list = []
        for cidx in stitched_coarse_indices:
            if cidx in coarse_part_nodes_map:
                stitched_nodes_list.append(coarse_part_nodes_map[cidx])
        
        if stitched_nodes_list:
            stitched_nodes = torch.cat(stitched_nodes_list)
            stitched_nodes = torch.unique(stitched_nodes)  # Remove duplicates
            
            # Step 4: Build stitched subgraph
            stitched_graph = _extract_subgraph(adj_t, stitched_nodes, original_data)
            
            if stitched_graph is not None and stitched_graph.num_nodes > 0:
                # Ensure stitched_graph has features
                if stitched_graph.x is None and original_data is not None:
                    if hasattr(stitched_graph, 'global_id') and stitched_graph.global_id is not None:
                        gids = stitched_graph.global_id.to(original_data.x.device)
                        stitched_graph.x = original_data.x[gids]
                
                stitched_global_ids = stitched_graph.global_id if hasattr(stitched_graph, 'global_id') else stitched_nodes
                
                # Step 5: VF2 on stitched graph
                vf2_stitched = vf2_verify_subgraph(
                    query_data, stitched_graph,
                    query_global_ids, stitched_global_ids,
                    timeout_seconds=vf2_timeout
                )
                result['vf2_stitched_found'] = vf2_stitched.found
                result['vf2_stitched_solutions'] = vf2_stitched.num_solutions
                result['vf2_stitched_time'] = vf2_stitched.latency_seconds
                result['stitched_nodes'] = len(stitched_nodes)
            else:
                result['vf2_stitched_found'] = False
                result['vf2_stitched_time'] = 0
        else:
            result['vf2_stitched_found'] = False
            result['vf2_stitched_time'] = 0
        
        # 4b. VF2 on FULL GRAPH baseline (Very slow! Optional)
        if run_baseline and original_data is not None:
            full_global_ids = torch.arange(original_data.num_nodes)
            vf2_baseline = vf2_verify_subgraph(
                query_data, original_data,
                query_global_ids, full_global_ids,
                timeout_seconds=vf2_timeout * 3  # More time for full graph
            )
            result['vf2_baseline_found'] = vf2_baseline.found
            result['vf2_baseline_solutions'] = vf2_baseline.num_solutions
            result['vf2_baseline_time'] = vf2_baseline.latency_seconds
        
        result['vf2_time'] = time.time() - t_vf2
        
        # 5. Compute precision/recall/F1 (predicted partition vs query nodes)
        predicted_nodes = set(predicted_global_ids.tolist()) if predicted_graph else set()
        ground_truth_nodes = set(query_global_ids.tolist())
        
        precision, recall, f1 = compute_precision_recall_f1(predicted_nodes, ground_truth_nodes)
        result['precision'] = precision
        result['recall'] = recall
        result['f1'] = f1
        
        result['success'] = True
        result['total_time'] = time.time() - total_start
        
    except Exception as e:
        result['error'] = str(e)
        result['total_time'] = time.time() - total_start
    
    return result


# =============================================================================
# STRATIFIED EVALUATION
# =============================================================================

def run_stratified_evaluation(
    encoder: ImprovedSubgraphEncoder,
    context: Dict[str, Any],
    query_types: List[str] = ['k_hop', 'single', 'multi_coarse'],
    queries_per_partition: int = 10,
    top_k: int = 5,
    device: torch.device = None,
    checkpoint_path: str = None  # NEW: Path for resumability
) -> pd.DataFrame:
    """
    Run stratified evaluation across all coarse partitions.
    
    Args:
        encoder: Trained encoder model
        context: Dict with graph data, hierarchies, FAISS indices
        query_types: List of query types to evaluate
        queries_per_partition: Queries to generate per partition per type
        top_k: K for recall@K metrics
        device: Torch device
        
    Returns:
        DataFrame with all results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder.eval()
    
    # Load checkpoint if resuming
    completed_keys = set()
    all_results = []
    if checkpoint_path:
        completed_keys, all_results = load_checkpoint(checkpoint_path)
    
    num_coarse = len(context['coarse_graphs'])
    original_data = context['original_data']
    adj_t = context['adj_t']
    
    # Query generators (using consolidated query_generator module)
    generators = {
        'k_hop': generate_k_hop_query,
        'single': generate_single_partition_query,
        'sibling_walk': generate_multi_fine_partition_query,
        'multi_coarse': generate_multi_coarse_partition_query,
    }
    
    for query_type in query_types:
        if query_type not in generators:
            print(f"[WARN] Unknown query type: {query_type}")
            continue
        
        generator = generators[query_type]
        print(f"\n{'='*60}")
        print(f"Evaluating query type: {query_type.upper()}")
        print(f"{'='*60}")
        
        pbar = tqdm(range(num_coarse), desc=f"{query_type}", unit="partition")
        
        for anchor_coarse_idx in pbar:
            # Check if this partition is already done for this query type
            partition_key = (query_type, anchor_coarse_idx)
            if partition_key in completed_keys:
                continue  # Skip completed partition
            
            for i in range(queries_per_partition):
                try:
                    # Generate query using consolidated generators (kwargs interface)
                    gen_kwargs = {
                        'original_data': original_data,
                        'adj_t': adj_t,
                        'device': device,
                        'min_nodes': QUERY_NODES_MIN,
                        'max_nodes': QUERY_NODES_MAX,
                        'fine_graphs': context.get('fine_graphs', []),
                        'fine_part_nodes_map': context.get('fine_part_nodes_map', {}),
                        'fine_to_coarse_map': context.get('fine_to_coarse_map', {}),
                        'coarse_to_fine_map': context.get('coarse_to_fine_map', {}),
                        'coarse_part_nodes_map': context.get('coarse_part_nodes_map', {}),
                        'coarse_part_graph': context.get('coarse_part_graph', None),
                        'node_to_coarse_map': context.get('node_to_coarse_map', {}),
                        'anchor_coarse_idx': anchor_coarse_idx,
                    }
                    
                    res = generator(**gen_kwargs)
                    
                    if res is None:
                        continue
                    
                    query_data, target_data, query_global_ids, true_coarse = res[:4]
                    
                    # Search and verify with ground truth target
                    result = search_and_verify(
                        query_data, query_global_ids, true_coarse,
                        context, encoder, device, top_k,
                        ground_truth_target=target_data,  # Stitched ground truth
                        run_baseline=False  # Set True for full graph comparison (SLOW)
                    )
                    
                    result['query_type'] = query_type
                    result['anchor_coarse'] = anchor_coarse_idx
                    result['query_id'] = f"{query_type}_{anchor_coarse_idx}_{i}"
                    
                    all_results.append(result)
                    
                except Exception as e:
                    all_results.append({
                        'success': False,
                        'error': str(e),
                        'query_type': query_type,
                        'anchor_coarse': anchor_coarse_idx
                    })
            
            # Mark partition complete and save checkpoint
            completed_keys.add(partition_key)
            if checkpoint_path:
                save_checkpoint(all_results, checkpoint_path, completed_keys)
    
    return pd.DataFrame(all_results)


def print_evaluation_summary(df: pd.DataFrame):
    """Print formatted evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    for query_type in df['query_type'].unique():
        subset = df[df['query_type'] == query_type]
        successful = subset[subset['success'] == True]
        
        print(f"\n--- {query_type.upper()} ---")
        print(f"  Total queries:       {len(subset)}")
        print(f"  Successful:          {len(successful)}")
        
        if len(successful) > 0:
            # Retrieval metrics
            print(f"  Coarse Recall@1:     {successful['coarse_correct'].mean()*100:.1f}%")
            print(f"  Coarse Recall@K:     {successful['coarse_in_top_k'].mean()*100:.1f}%")
            
            # Partition coverage metrics (KEY)
            if 'gt_partition_recall' in successful.columns:
                gt_recall = successful['gt_partition_recall'].mean()*100
                print(f"  GT Partition Recall: {gt_recall:.1f}%")
            if 'all_gt_partitions_found' in successful.columns:
                all_found = successful['all_gt_partitions_found'].mean()*100
                print(f"  All GT Parts Found:  {all_found:.1f}%")
            
            # VF2 metrics (2-way comparison)
            if 'vf2_stitched_found' in successful.columns:
                vf2_stitched = successful['vf2_stitched_found'].mean()*100
                print(f"  VF2 Stitched Match:  {vf2_stitched:.1f}% (Our Method)")
            if 'vf2_baseline_found' in successful.columns:
                vf2_base = successful[successful['vf2_baseline_found'].notna()]['vf2_baseline_found'].mean()*100
                print(f"  VF2 Baseline Match:  {vf2_base:.1f}% (Full Graph)")
            
            # Node counts
            if 'query_nodes' in successful.columns:
                print(f"  Avg Query Nodes:     {successful['query_nodes'].mean():.0f}")
            if 'stitched_nodes' in successful.columns:
                print(f"  Avg Stitched Nodes:  {successful['stitched_nodes'].mean():.0f}")
            if 'num_stitched' in successful.columns:
                print(f"  Avg Partitions:      {successful['num_stitched'].mean():.1f}")
            
            # Timing
            print(f"  Avg Latency:         {successful['total_time'].mean()*1000:.1f}ms")
            print(f"    - Embedding:       {successful['embed_time'].mean()*1000:.1f}ms")
            print(f"    - FAISS:           {successful['faiss_time'].mean()*1000:.1f}ms")
            print(f"    - VF2:             {successful['vf2_time'].mean()*1000:.1f}ms")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run evaluation on specified dataset."""
    import argparse
    import torch.nn as nn
    
    parser = argparse.ArgumentParser(description='Evaluate Jigsaw GNN')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'arxiv', 'mag'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--queries_per_partition', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--output', type=str, default='evaluation_results.csv')
    parser.add_argument('--hierarchy_cache', type=str, default=None,
                        help='Path to cache/load hierarchy pickle (e.g., cache/cora_hierarchy.pkl)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint JSON for resumability')
    
    args = parser.parse_args()
    
    print(f"[INFO] Evaluating on {args.dataset}")
    print(f"[INFO] Device: {DEVICE}")
    
    # Load data and model
    from src.data import load_dataset, get_or_build_hierarchy
    from src.model import NodeFeatureAugmentor
    
    data = load_dataset(args.dataset)
    
    # MAG uses NodeFeatureAugmentor, Cora/Arxiv don't
    TYPE_DIM = 16
    NODE_DIM = 16
    
    if args.dataset == 'mag':
        print("[INFO] Initializing NodeFeatureAugmentor for MAG...")
        augmentor = NodeFeatureAugmentor(
            num_nodes=data.num_nodes, 
            num_types=len(data.node_types), 
            type_dim=TYPE_DIM, 
            node_dim=NODE_DIM
        ).to(DEVICE)
        base_feat_dim = data.x.size(1)
        augmented_feat_dim = base_feat_dim + augmentor.added_dim
    else:
        print(f"[INFO] Skipping NodeFeatureAugmentor for {args.dataset}...")
        augmentor = None  # No augmentor for Cora/Arxiv
        base_feat_dim = data.x.size(1)
        augmented_feat_dim = base_feat_dim
    
    print(f"[INFO] Base features: {base_feat_dim}, Model input dim: {augmented_feat_dim}")
    
    encoder = ImprovedSubgraphEncoder(
        in_neurons=augmented_feat_dim,
        hidden_neurons=GIN_HIDDEN_NEURONS,
        output_neurons=GIN_OUTPUT_NEURONS
    ).to(DEVICE)
    
    if args.model_path and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        
        # Load encoder
        if 'encoder' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder'])
        elif 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            # Assume the checkpoint IS the encoder state_dict
            encoder.load_state_dict(checkpoint)
        
        # Load augmentor for MAG
        if args.dataset == 'mag' and augmentor is not None:
            if 'augmentor' in checkpoint:
                augmentor.load_state_dict(checkpoint['augmentor'])
            elif 'augmentor_state_dict' in checkpoint:
                augmentor.load_state_dict(checkpoint['augmentor_state_dict'])
            else:
                print("[WARN] MAG checkpoint missing augmentor weights!")
        
        print(f"[INFO] Loaded model from {args.model_path}")
    
    encoder.eval()
    if augmentor is not None:
        augmentor.eval()
    
    # Build hierarchy and context (with optional caching)
    cfg = PARTITION_CONFIGS.get(args.dataset, PARTITION_CONFIGS['default'])
    hierarchy = get_or_build_hierarchy(data, cfg['coarse'], cfg['fine'], args.hierarchy_cache)
    
    # Build FAISS index (need to handle augmentor for MAG)
    print("[INFO] Building FAISS index...")
    coarse_graphs = hierarchy['coarse_graphs']
    
    # Helper to get embedding with optional augmentor
    # NOTE: Hierarchy graphs have x=None (keep_features=False for RAM saving)
    # We fetch features from original data using global_id
    def embed_with_augmentor(g, encoder, device, augmentor=None, original_data=None):
        """Get embedding, fetching features from original data if needed."""
        if g is None or g.num_nodes == 0:
            return torch.zeros(1, GIN_OUTPUT_NEURONS, device=device)
        
        g = g.to(device)
        
        # If features are missing, fetch from original data using global_id
        if g.x is None and original_data is not None:
            if hasattr(g, 'global_id') and g.global_id is not None:
                global_ids = g.global_id.to(original_data.x.device)
                g.x = original_data.x[global_ids].to(device)
            else:
                # Fallback: can't get features
                print(f"[WARN] Graph missing x and global_id, using zeros")
                g.x = torch.zeros(g.num_nodes, original_data.x.size(1), device=device)
        
        if augmentor is not None:
            g = g.clone()
            g.x = augmentor(g)
        
        return get_graph_embedding(g, encoder, device)
    
    # Embed coarse graphs with progress bar (this can be slow on CPU)
    print(f"[INFO] Embedding {len([g for g in coarse_graphs if g is not None])} coarse partitions...")
    coarse_embeds_list = []
    for i, g in enumerate(tqdm(coarse_graphs, desc="Embedding coarse partitions")):
        if g is not None:
            emb = embed_with_augmentor(g, encoder, DEVICE, augmentor, data)
            coarse_embeds_list.append(emb)
    coarse_embeds = torch.cat(coarse_embeds_list, dim=0)
    
    faiss_coarse = faiss.IndexFlatL2(GIN_OUTPUT_NEURONS)
    faiss_coarse.add(coarse_embeds.cpu().numpy())
    
    # Build context
    adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    
    context = {
        'original_data': data,
        'adj_t': adj_t,
        'coarse_graphs': coarse_graphs,
        'fine_graphs': hierarchy['fine_graphs'],
        'coarse_part_nodes_map': hierarchy['coarse_part_nodes_map'],
        'fine_part_nodes_map': hierarchy['fine_part_nodes_map'],
        'fine_to_coarse_map': hierarchy['fine_to_coarse_map'],
        'coarse_part_graph': hierarchy['coarse_part_graph'],
        'node_to_coarse_map': hierarchy['node_to_coarse_map'],
        'faiss_coarse': faiss_coarse,
        'augmentor': augmentor,  # Pass augmentor for MAG (None for others)
    }
    
    # Build coarse_to_fine_map
    coarse_to_fine = defaultdict(list)
    for f, c in hierarchy['fine_to_coarse_map'].items():
        coarse_to_fine[c].append(f)
    context['coarse_to_fine_map'] = dict(coarse_to_fine)
    
    # Run evaluation (with optional checkpointing)
    df = run_stratified_evaluation(
        encoder, context,
        query_types=['k_hop', 'single', 'sibling_walk', 'multi_coarse'],
        queries_per_partition=args.queries_per_partition,
        top_k=args.top_k,
        device=DEVICE,
        checkpoint_path=args.checkpoint  # NEW: enable resumability
    )
    
    # Save and print results
    df.to_csv(args.output, index=False)
    print(f"\n[INFO] Results saved to {args.output}")
    
    print_evaluation_summary(df)


if __name__ == '__main__':
    main()
