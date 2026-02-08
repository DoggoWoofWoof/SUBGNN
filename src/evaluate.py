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
    PARTITION_CONFIGS,
    BASELINE_SAMPLE_SIZE, BASELINE_TIMEOUT_SECONDS, BASELINE_MAX_RETRIES,
    TARGET_QUERIES_PER_TYPE
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
# RANDOM SAMPLING BASELINE
# =============================================================================

def run_random_sampling_baseline(
    query_data: Data,
    query_global_ids: torch.Tensor,
    original_data: Data,
    adj_t: SparseTensor,
    sample_size: int = BASELINE_SAMPLE_SIZE,
    timeout_per_attempt: float = BASELINE_TIMEOUT_SECONDS,
    max_retries: int = BASELINE_MAX_RETRIES
) -> Dict[str, Any]:
    """
    K-hop expansion baseline for big graph search (SOTA approach).
    
    Instead of searching the entire graph (infeasible for large graphs),
    pick a random anchor and expand via k-hop to get ~sample_size connected nodes.
    Run VF2 on this connected neighborhood. Retry with different anchors.
    
    This is the fair baseline for comparing FAISS-based search with brute-force.
    
    Args:
        query_data: Query subgraph
        query_global_ids: Global node IDs in query
        original_data: Full graph
        adj_t: Sparse adjacency tensor
        sample_size: Target nodes per sample (via k-hop expansion)
        timeout_per_attempt: Timeout in seconds per VF2 attempt
        max_retries: Max number of random anchor attempts
        
    Returns:
        Dict with found, total_time, attempts, etc.
    """
    result = {
        'baseline_found': False,
        'baseline_attempts': 0,
        'baseline_time': 0.0,
        'baseline_sample_size': sample_size,
        'baseline_timeout_per_attempt': timeout_per_attempt
    }
    
    num_nodes = original_data.num_nodes
    total_start = time.time()
    
    for attempt in range(max_retries):
        result['baseline_attempts'] = attempt + 1
        elapsed = time.time() - total_start
        print(f"      [Baseline] Try {attempt+1}/{max_retries} ({elapsed:.0f}s elapsed)", flush=True)
        
        # Pick random anchor and expand via k-hop
        anchor = random.randint(0, num_nodes - 1)
        
        # Start with small k and increase until we hit target size
        for k in range(1, 10):  # Max 10 hops
            try:
                subset, _, _, _ = k_hop_subgraph(
                    node_idx=anchor,
                    num_hops=k,
                    edge_index=original_data.edge_index,
                    relabel_nodes=False,
                    num_nodes=num_nodes
                )
                
                if len(subset) >= sample_size:
                    # Trim to sample_size if too large
                    if len(subset) > sample_size:
                        # Take closest nodes via BFS order (subset is already BFS-ordered)
                        subset = subset[:sample_size]
                    break
            except Exception:
                continue
        else:
            # Couldn't get enough nodes, use what we have
            if len(subset) < 10:
                print(f"      [Baseline] Skip: too few nodes ({len(subset)})", flush=True)
                continue  # Too small, try another anchor
        
        # Build sampled subgraph
        sampled_graph = _extract_subgraph(adj_t, subset, original_data)
        
        if sampled_graph is None or sampled_graph.num_nodes == 0:
            print(f"      [Baseline] Skip: empty subgraph", flush=True)
            continue
        
        print(f"      [Baseline] Searching {len(subset)} nodes (k={k}, timeout={timeout_per_attempt:.0f}s)...", flush=True)
        
        # Run VF2 on sampled subgraph
        try:
            vf2_result = vf2_verify_subgraph(
                query_data, sampled_graph,
                query_global_ids, subset,
                timeout_seconds=timeout_per_attempt
            )
            
            if vf2_result.found:
                result['baseline_found'] = True
                result['baseline_time'] = time.time() - total_start
                result['baseline_solutions'] = vf2_result.num_solutions
                result['baseline_k_hops'] = k
                return result
                
        except Exception as e:
            # Timeout or other error, continue to next attempt
            pass
        
        # Check total time limit
        if time.time() - total_start > max_retries * timeout_per_attempt:
            break
    
    result['baseline_time'] = time.time() - total_start
    return result



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
    top_k: int = 20,  # Changed from 5 to 20 for full partition coverage
    vf2_timeout: float = 30.0,
    ground_truth_target: Optional[Data] = None,  # Stitched ground truth target
    run_baseline: bool = False,  # Whether to run VF2 on full graph (SLOW)
    skip_vf2: bool = False  # Skip VF2 verification for faster testing
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
        print("      [S&V] Step 1: Embedding query...", flush=True)
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
        print(f"      [S&V] Step 1 done ({result['embed_time']:.2f}s)", flush=True)
        
        # 2. Coarse FAISS search
        print("      [S&V] Step 2: Coarse FAISS...", flush=True)
        t_faiss = time.time()
        faiss_coarse = context['faiss_coarse']
        num_coarse = context.get('num_coarse', 20)
        faiss_idx_to_coarse_id = context.get('faiss_idx_to_coarse_id', {})
        
        # Search for ALL partitions to compute accurate recall@k
        search_k = min(num_coarse, 20)  # At least 20, or all if fewer
        D_coarse, I_coarse = faiss_coarse.search(zq.cpu().numpy(), search_k)
        
        # CRITICAL: Translate FAISS indices to actual coarse partition IDs
        # FAISS returns indices in its own index space, not actual coarse partition IDs
        if faiss_idx_to_coarse_id:
            I_coarse_translated = [faiss_idx_to_coarse_id.get(int(idx), int(idx)) for idx in I_coarse[0]]
        else:
            I_coarse_translated = [int(idx) for idx in I_coarse[0]]
        
        predicted_coarse_idx = I_coarse_translated[0]
        result['predicted_coarse'] = int(predicted_coarse_idx)
        result['coarse_correct'] = predicted_coarse_idx in true_coarse_indices
        result['coarse_in_top_k'] = any(idx in true_coarse_indices for idx in I_coarse_translated[:top_k])
        # Compute recall: how many ground truth partitions are in top-k
        gt_in_topk = sum(1 for idx in I_coarse_translated[:top_k] if idx in true_coarse_indices)
        result['coarse_recall_at_k'] = gt_in_topk / len(true_coarse_indices) if true_coarse_indices else 0
        result['faiss_time'] = time.time() - t_faiss
        print(f"      [S&V] Step 2 done ({result['faiss_time']:.2f}s)", flush=True)
        
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
            
            print(f"      [S&V] Step 3: Embedding {len(candidate_fines)} fine graphs...", flush=True)
            candidate_embeds = torch.cat([embed_fine(fine_graphs[i]) for i in candidate_fines], dim=0)
            print(f"      [S&V] Step 3 done", flush=True)
            
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
        # ================================================================
        # 4a. ITERATIVE STITCHING: Start small, expand on VF2 failure
        #     Step 1: Try just top-1 partition
        #     Step 2: If fail, expand to top-5 neighbors  
        #     Step 3: If still fail, expand to top-20
        # ================================================================
        print("      [S&V] Step 4: Iterative VF2...", flush=True)
        t_vf2 = time.time()
        
        coarse_part_nodes_map = context['coarse_part_nodes_map']
        coarse_part_graph = context.get('coarse_part_graph')
        adj_t = context['adj_t']
        
        # Log partition info (use translated indices!)
        faiss_top20 = I_coarse_translated[:20]  # Already translated above
        print(f"      [S&V] True partitions: {sorted(list(true_coarse_indices))}", flush=True)
        print(f"      [S&V] FAISS top-20: {faiss_top20}", flush=True)
        
        # Calculate recall at different levels
        rec1 = sum(1 for gt in true_coarse_indices if gt in faiss_top20[:1]) / len(true_coarse_indices) if true_coarse_indices else 0
        rec5 = sum(1 for gt in true_coarse_indices if gt in faiss_top20[:5]) / len(true_coarse_indices) if true_coarse_indices else 0
        rec20 = sum(1 for gt in true_coarse_indices if gt in faiss_top20[:20]) / len(true_coarse_indices) if true_coarse_indices else 0
        print(f"      [S&V] Recall@1={rec1*100:.0f}% @5={rec5*100:.0f}% @20={rec20*100:.0f}%", flush=True)
        
        # Store level-specific recall metrics in result
        result['recall_at_1'] = rec1
        result['recall_at_5'] = rec5
        result['recall_at_20'] = rec20
        
        # VF2 Configuration: 3 expansion levels, 60s per level (180s total = 3 min)
        # Note: final_stitched_indices will be set to faiss_top20 as fallback AFTER VF2 attempts
        vf2_found = False
        vf2_solutions = 0
        vf2_time = 0.0
        stitched_nodes_count = 0
        final_stitched_indices = faiss_top20[:20]  # Fallback to FAISS if VF2 fails
        vf2_level_reached = "none"  # Track which level VF2 succeeded/failed at
        
        # Define expansion levels: (num_partitions, description)
        # Note: VF2 call is skipped inside the loop if skip_vf2=True
        expansion_levels = [
            (1, "top-1"),
            (5, "top-5"),
            (20, "top-20"),
        ]
        
        for max_parts, level_name in expansion_levels:
            # Build stitched indices for this level
            if coarse_part_graph is not None:
                # Get top partitions and their neighbors (use TRANSLATED indices!)
                top_indices = set(int(idx) for idx in I_coarse_translated[:min(max_parts, top_k)])
                
                if max_parts > 1:
                    # Add neighbors of top partitions
                    expanded = set(top_indices)
                    for cidx in top_indices:
                        if coarse_part_graph.has_node(cidx):
                            neighbors = list(coarse_part_graph.neighbors(cidx))
                            expanded.update(neighbors)
                    
                    # Limit to max_parts, prioritize FAISS-ranked (use TRANSLATED indices!)
                    faiss_ranked = [int(idx) for idx in I_coarse_translated[:top_k]]
                    stitched_coarse_indices = []
                    for idx in faiss_ranked:
                        if idx in expanded:
                            stitched_coarse_indices.append(idx)
                            if len(stitched_coarse_indices) >= max_parts:
                                break
                    for idx in expanded:
                        if idx not in stitched_coarse_indices:
                            stitched_coarse_indices.append(idx)
                            if len(stitched_coarse_indices) >= max_parts:
                                break
                else:
                    stitched_coarse_indices = list(top_indices)
            else:
                # Fallback without coarse_part_graph (use TRANSLATED indices!)
                stitched_coarse_indices = [int(idx) for idx in I_coarse_translated[:max_parts]]
            
            # Build stitched subgraph
            stitched_nodes_list = []
            for cidx in stitched_coarse_indices:
                if cidx in coarse_part_nodes_map:
                    stitched_nodes_list.append(coarse_part_nodes_map[cidx])
            
            if not stitched_nodes_list:
                continue
                
            stitched_nodes = torch.cat(stitched_nodes_list)
            stitched_nodes = torch.unique(stitched_nodes)
            stitched_graph = _extract_subgraph(adj_t, stitched_nodes, original_data)
            
            if stitched_graph is None or stitched_graph.num_nodes == 0:
                continue
            
            # Ensure features
            if stitched_graph.x is None and original_data is not None:
                if hasattr(stitched_graph, 'global_id') and stitched_graph.global_id is not None:
                    gids = stitched_graph.global_id.to(original_data.x.device)
                    stitched_graph.x = original_data.x[gids]
            
            stitched_global_ids = stitched_graph.global_id if hasattr(stitched_graph, 'global_id') else stitched_nodes
            
            # Update stitching metrics (always computed, regardless of skip_vf2)
            stitched_nodes_count = len(stitched_nodes)
            final_stitched_indices = stitched_coarse_indices
            vf2_level_reached = level_name  # Track which level we're at
            
            # VF2 on this level (skip if skip_vf2 is True)
            if skip_vf2:
                print(f"      [S&V] Expansion {level_name}: Q={query_data.num_nodes}, S={stitched_graph.num_nodes} nodes (VF2 skipped)", flush=True)
                # Don't run VF2, just continue to next level to get max stitching
            else:
                level_timeout = 60.0  # 60s per level = 3 min total across 3 levels
                print(f"      [S&V] VF2 {level_name}: Q={query_data.num_nodes}, S={stitched_graph.num_nodes} nodes...", flush=True)
                
                vf2_result = vf2_verify_subgraph(
                    query_data, stitched_graph,
                    query_global_ids, stitched_global_ids,
                    timeout_seconds=level_timeout
                )
                
                vf2_time += vf2_result.latency_seconds
                
                if vf2_result.found:
                    vf2_found = True
                    vf2_solutions = vf2_result.num_solutions
                    print(f"      [S&V] VF2 {level_name}: FOUND! ({vf2_result.latency_seconds:.1f}s)", flush=True)
                    break
                else:
                    print(f"      [S&V] VF2 {level_name}: not found ({vf2_result.latency_seconds:.1f}s), expanding...", flush=True)
        
        result['vf2_stitched_found'] = vf2_found
        result['vf2_stitched_solutions'] = vf2_solutions
        result['vf2_stitched_time'] = vf2_time
        result['vf2_level_reached'] = vf2_level_reached  # Which expansion level VF2 stopped at
        result['stitched_nodes'] = stitched_nodes_count
        result['stitched_partitions'] = final_stitched_indices
        result['num_stitched'] = len(final_stitched_indices)
        
        # KEY METRIC: Are ground truth partitions covered by final stitched?
        gt_partitions_in_stitched = sum(1 for gt in true_coarse_indices if gt in final_stitched_indices)
        result['gt_partitions_covered'] = gt_partitions_in_stitched
        result['gt_partitions_total'] = len(true_coarse_indices)
        result['gt_partition_recall'] = gt_partitions_in_stitched / len(true_coarse_indices) if true_coarse_indices else 0
        result['all_gt_partitions_found'] = (gt_partitions_in_stitched == len(true_coarse_indices))
        
        # Debug: why would recall be 0 with 18k nodes?
        print(f"      [S&V] Final: stitched_parts={final_stitched_indices[:5]}... ({len(final_stitched_indices)} total)", flush=True)
        print(f"      [S&V] Final: true_parts={list(true_coarse_indices)[:5]}... ({len(true_coarse_indices)} total)", flush=True)
        print(f"      [S&V] Final: gt_in_stitched={gt_partitions_in_stitched}/{len(true_coarse_indices)} = {result['gt_partition_recall']*100:.0f}%", flush=True)
        
        # Query stats
        result['query_nodes'] = len(query_global_ids) if query_global_ids is not None else 0
        
        # 4b. Random Sampling Baseline (for big graphs)
        if run_baseline and original_data is not None:
            baseline_result = run_random_sampling_baseline(
                query_data, query_global_ids,
                original_data, adj_t,
                sample_size=BASELINE_SAMPLE_SIZE,
                timeout_per_attempt=BASELINE_TIMEOUT_SECONDS,
                max_retries=BASELINE_MAX_RETRIES
            )
            result['vf2_baseline_found'] = baseline_result['baseline_found']
            result['vf2_baseline_time'] = baseline_result['baseline_time']
            result['vf2_baseline_attempts'] = baseline_result['baseline_attempts']
            result['vf2_baseline_sample_size'] = baseline_result['baseline_sample_size']
        
        result['vf2_time'] = time.time() - t_vf2
        
        # 5. Compute precision/recall/F1 (stitched partition nodes vs query nodes)
        # Note: stitched_nodes are the nodes from FAISS-predicted partitions
        if 'stitched_nodes' in result and result['stitched_nodes'] > 0:
            # Handle both tensor and list cases
            if hasattr(stitched_nodes, 'tolist'):
                predicted_nodes = set(stitched_nodes.tolist())
            elif isinstance(stitched_nodes, list):
                predicted_nodes = set(stitched_nodes)
            else:
                predicted_nodes = set()
        else:
            predicted_nodes = set()
        
        # Handle both tensor and list for query_global_ids
        if query_global_ids is not None:
            if hasattr(query_global_ids, 'tolist'):
                ground_truth_nodes = set(query_global_ids.tolist())
            elif isinstance(query_global_ids, list):
                ground_truth_nodes = set(query_global_ids)
            else:
                ground_truth_nodes = set()
        else:
            ground_truth_nodes = set()
        
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
    queries_per_partition: int = None,  # If None, auto-calculate from target
    target_queries_per_type: int = TARGET_QUERIES_PER_TYPE,  # Total queries per type
    top_k: int = 5,
    device: torch.device = None,
    checkpoint_path: str = None,
    run_baseline: bool = False,  # Whether to run k-hop baseline comparison
    skip_vf2: bool = False  # Skip VF2 verification for faster testing
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
    
    # Auto-calculate queries_per_partition if not specified
    if queries_per_partition is None:
        queries_per_partition = max(1, target_queries_per_type // num_coarse)
        print(f"[INFO] Auto-calculated: {queries_per_partition} queries/partition x {num_coarse} partitions = ~{queries_per_partition * num_coarse} queries/type")
    
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
                print(f"  [P{anchor_coarse_idx}] Generating query {i}...", flush=True)
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
                        'G_nx': context.get('G_nx', None),  # For sibling_walk and multi_coarse 
                        'anchor_coarse_idx': anchor_coarse_idx,
                        'num_frags': 2,  # For sibling_walk and multi_fine (reduced for small graphs)
                    }
                    
                    res = generator(**gen_kwargs)
                    
                    if res is None:
                        print(f"    [P{anchor_coarse_idx}][Q{i}] Query gen failed", flush=True)
                        continue
                    
                    # Query generators return 5 elements: (query, target, global_ids, true_fine, true_coarse)
                    # Note: res[3] is true_fine_idx, res[4] is true_coarse_idx (set)
                    query_data, target_data, query_global_ids = res[0], res[1], res[2]
                    true_coarse = res[4] if len(res) > 4 else res[3]  # Some generators return 4, some return 5
                    print(f"    [P{anchor_coarse_idx}][Q{i}] Generated: Q={query_data.num_nodes} nodes, T={target_data.num_nodes if target_data else 0} nodes", flush=True)
                    
                    # Search and verify with ground truth target  
                    print(f"    [P{anchor_coarse_idx}][Q{i}] Starting search_and_verify...", flush=True)
                    result = search_and_verify(
                        query_data, query_global_ids, true_coarse,
                        context, encoder, device, top_k,
                        ground_truth_target=target_data,
                        run_baseline=run_baseline,  # Run k-hop baseline if enabled
                        skip_vf2=skip_vf2  # Skip VF2 if enabled
                    )
                    print(f"    [P{anchor_coarse_idx}][Q{i}] search_and_verify done (success={result.get('success', False)})", flush=True)
                    
                    result['query_type'] = query_type
                    result['anchor_coarse'] = anchor_coarse_idx
                    result['query_id'] = f"{query_type}_{anchor_coarse_idx}_{i}"
                    
                    # Useful logging: query size, stitched size, recall, VF2 results
                    q_nodes = result.get('query_nodes', 0)
                    stitched = result.get('stitched_nodes', 0)
                    faiss_found = "✓" if result.get('vf2_stitched_found', False) else "✗"
                    recall = result.get('gt_partition_recall', 0) * 100
                    
                    # Build log string
                    log_str = f"    [P{anchor_coarse_idx}][Q{i}] Q:{q_nodes} → Stitch:{stitched} (rec:{recall:.0f}%) FAISS:{faiss_found}"
                    
                    # Add baseline result if enabled
                    if 'vf2_baseline_found' in result:
                        baseline_found = "✓" if result.get('vf2_baseline_found', False) else "✗"
                        baseline_tries = result.get('vf2_baseline_attempts', 0)
                        log_str += f" | Baseline:{baseline_found}({baseline_tries}tries)"
                    
                    print(log_str, flush=True)
                    
                    all_results.append(result)
                    
                except Exception as e:
                    import traceback
                    print(f"    [P{anchor_coarse_idx}][Q{i}] ERROR: {e}", flush=True)
                    traceback.print_exc()
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
    
    # Separate actual queries from summary rows
    actual_queries = df[~df['query_id'].str.contains('_MEDIAN|_AVERAGE', na=False)]
    
    # Print per-query-type summaries (excluding OVERALL which is a summary, not a query type)
    query_types = [qt for qt in df['query_type'].unique() if qt != 'OVERALL']
    
    for query_type in query_types:
        subset = actual_queries[actual_queries['query_type'] == query_type]
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
    
    # Print OVERALL summary computed from ALL actual queries
    all_successful = actual_queries[actual_queries['success'] == True]
    print(f"\n--- OVERALL ---")
    print(f"  Total queries:       {len(actual_queries)}")
    print(f"  Successful:          {len(all_successful)}")
    
    if len(all_successful) > 0:
        print(f"  Coarse Recall@1:     {all_successful['coarse_correct'].mean()*100:.1f}%")
        print(f"  Coarse Recall@K:     {all_successful['coarse_in_top_k'].mean()*100:.1f}%")
        if 'gt_partition_recall' in all_successful.columns:
            print(f"  GT Partition Recall: {all_successful['gt_partition_recall'].mean()*100:.1f}%")
        if 'all_gt_partitions_found' in all_successful.columns:
            print(f"  All GT Parts Found:  {all_successful['all_gt_partitions_found'].mean()*100:.1f}%")
        if 'vf2_stitched_found' in all_successful.columns:
            print(f"  VF2 Stitched Match:  {all_successful['vf2_stitched_found'].mean()*100:.1f}% (Our Method)")
        if 'query_nodes' in all_successful.columns:
            print(f"  Avg Query Nodes:     {all_successful['query_nodes'].mean():.0f}")
        if 'stitched_nodes' in all_successful.columns:
            print(f"  Avg Stitched Nodes:  {all_successful['stitched_nodes'].mean():.0f}")
        print(f"  Avg Latency:         {all_successful['total_time'].mean()*1000:.1f}ms")


def add_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary rows (median, average) per query type and overall to the DataFrame.
    """
    summary_rows = []
    
    # Key metrics to summarize
    numeric_cols = ['recall_at_1', 'recall_at_5', 'recall_at_20', 'gt_partition_recall',
                    'coarse_recall_at_k', 'embed_time', 'faiss_time', 'vf2_time', 'total_time',
                    'query_nodes', 'stitched_nodes', 'num_stitched']
    
    # Filter to only columns that exist
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    # Summary per query type
    if 'query_type' in df.columns:
        for qtype in df['query_type'].unique():
            subset = df[df['query_type'] == qtype]
            
            # Median row
            median_row = {'query_type': qtype, 'query_id': f'{qtype}_MEDIAN', 'success': True}
            for col in numeric_cols:
                if col in subset.columns:
                    median_row[col] = subset[col].median()
            summary_rows.append(median_row)
            
            # Average row
            avg_row = {'query_type': qtype, 'query_id': f'{qtype}_AVERAGE', 'success': True}
            for col in numeric_cols:
                if col in subset.columns:
                    avg_row[col] = subset[col].mean()
            summary_rows.append(avg_row)
    
    # Overall summary
    overall_median = {'query_type': 'OVERALL', 'query_id': 'OVERALL_MEDIAN', 'success': True}
    overall_avg = {'query_type': 'OVERALL', 'query_id': 'OVERALL_AVERAGE', 'success': True}
    for col in numeric_cols:
        if col in df.columns:
            overall_median[col] = df[col].median()
            overall_avg[col] = df[col].mean()
    summary_rows.append(overall_median)
    summary_rows.append(overall_avg)
    
    # Append summary rows to DataFrame
    summary_df = pd.DataFrame(summary_rows)
    df = pd.concat([df, summary_df], ignore_index=True)
    
    return df


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
    parser.add_argument('--queries_per_partition', type=int, default=None,
                        help='Queries per partition (if None, auto-calc from --target_queries)')
    parser.add_argument('--target_queries', type=int, default=TARGET_QUERIES_PER_TYPE,
                        help='Target total queries per query type (default: 100)')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--output', type=str, default='evaluation_results.csv')
    parser.add_argument('--hierarchy_cache', type=str, default=None,
                        help='Path to cache/load hierarchy pickle (e.g., cache/cora_hierarchy.pkl)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint JSON for resumability')
    parser.add_argument('--run_baseline', action='store_true',
                        help='Run k-hop random sampling baseline (SLOW, for comparison)')
    parser.add_argument('--skip_vf2', action='store_true',
                        help='Skip VF2 verification (faster, only FAISS metrics)')
    
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
    # CRITICAL: Track mapping from FAISS index to actual coarse partition ID
    # because coarse_graphs may have None entries that we skip
    print(f"[INFO] Embedding {len([g for g in coarse_graphs if g is not None])} coarse partitions...")
    coarse_embeds_list = []
    faiss_idx_to_coarse_id = {}  # Maps FAISS index -> actual coarse partition ID
    for coarse_id, g in enumerate(tqdm(coarse_graphs, desc="Embedding coarse partitions")):
        if g is not None:
            emb = embed_with_augmentor(g, encoder, DEVICE, augmentor, data)
            faiss_idx_to_coarse_id[len(coarse_embeds_list)] = coarse_id  # Map FAISS idx to coarse ID
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
        'faiss_idx_to_coarse_id': faiss_idx_to_coarse_id,  # CRITICAL: Maps FAISS results to actual coarse IDs
        'num_coarse': len(coarse_graphs),
        'augmentor': augmentor,  # Pass augmentor for MAG (None for others)
        'G_nx': to_networkx(data, to_undirected=True),  # For sibling_walk and multi_coarse
    }
    
    # Build coarse_to_fine_map
    coarse_to_fine = defaultdict(list)
    for f, c in hierarchy['fine_to_coarse_map'].items():
        coarse_to_fine[c].append(f)
    context['coarse_to_fine_map'] = dict(coarse_to_fine)
    
    # Run evaluation (with optional checkpointing)
    df = run_stratified_evaluation(
        encoder, context,
        query_types=['k_hop', 'single', 'sibling_walk', 'multi_coarse'],  # All query types
        queries_per_partition=args.queries_per_partition,
        target_queries_per_type=args.target_queries,
        top_k=args.top_k,
        device=DEVICE,
        checkpoint_path=args.checkpoint,
        run_baseline=args.run_baseline,
        skip_vf2=args.skip_vf2
    )
    
    # Add summary statistics (median, avg) per query type
    df = add_summary_rows(df)
    
    # Save and print results
    df.to_csv(args.output, index=False)
    print(f"\n[INFO] Results saved to {args.output}")
    
    print_evaluation_summary(df)


if __name__ == '__main__':
    main()
