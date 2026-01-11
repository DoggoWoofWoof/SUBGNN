"""
Search and Verification Module

Handles FAISS-based search and VF2/Glasgow verification.
"""

import os
import random
import re
import subprocess
import time
from typing import Any, Callable, Dict, Tuple, List

import faiss
import torch
from torch_geometric.data import Data

from src.config import FAISS_TOP_K, SOLVER_PATH
from src.model import ImprovedSubgraphEncoder, get_graph_embedding
from src.utils import convert_graph_to_csv


def _run_glasgow_solver(
    Gq,
    q_global_nodes: list,
    target_graph: Data,
    target_local_to_global_map: Dict[int, int],
    start_node_global_id: int,
    context: Dict[str, Any],
) -> Tuple[bool, int, float, float, float, float]:
    """
    Runs the Glasgow solver executable for subgraph isomorphism.
    
    Returns:
        (found, solution_count, first_accuracy, time_to_first, time_to_correct, time_to_all)
    """
    query_csv = context.get("query_csv_path", "query.csv")
    target_csv = context.get("target_csv_path", "target.csv")
    solver_path = context.get("solver_path", SOLVER_PATH)

    query_local_to_global_map = {i: g_id for i, g_id in enumerate(q_global_nodes)}
    
    # Export CSVs
    convert_graph_to_csv(Gq, query_csv, context["global_id_to_label_feature"], query_local_to_global_map)
    convert_graph_to_csv(target_graph, target_csv, context["global_id_to_label_feature"], target_local_to_global_map)

    # Check if solver exists
    if not os.path.exists(solver_path):
        print(f"[Warning] Solver binary not found at {solver_path}. Skipping exact verification.")
        return False, 0, -1, -1, -1, -1

    # Build command
    cmd = [
        solver_path,
        "--count-solutions",
        "--induced",
        "--print-all-solutions",
        query_csv,
        target_csv,
    ]

    # Regex patterns for parsing output
    mapping_pattern = re.compile(r"^mapping\s*=\s*(.*)$")
    solution_count_pattern = re.compile(r"^solution_count\s*=\s*(\d+)$")
    
    solution_found = False
    solution_count = 0
    first_accuracy = -1
    time_to_first = -1
    time_to_correct = -1
    time_to_all = -1
    
    start_time = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            line = line.strip()
            
            if mapping_match := mapping_pattern.match(line):
                now = time.time()
                if time_to_first < 0:
                    time_to_first = now - start_time
                
                solution_found = True
                
                # Parse mapping and compute accuracy
                pairs = re.findall(r"\((\d+)\s*->\s*(\d+)\)", mapping_match.group(1))
                mapping_dict = {int(q): int(t) for q, t in pairs}
                
                correct_nodes = 0
                for q_local, t_local in mapping_dict.items():
                    q_gid = query_local_to_global_map.get(q_local)
                    t_gid = target_local_to_global_map.get(t_local)
                    if q_gid == t_gid:
                        correct_nodes += 1
                
                acc = (correct_nodes / len(mapping_dict)) * 100 if mapping_dict else 0
                if first_accuracy < 0:
                    first_accuracy = acc
                if acc == 100 and time_to_correct < 0:
                    time_to_correct = now - start_time

            elif count_match := solution_count_pattern.match(line):
                solution_count = int(count_match.group(1))
                time_to_all = time.time() - start_time
                
        proc.wait()
    except Exception as e:
        print(f"[Solver Error] {e}")
        return False, 0, 0, 0, 0, 0

    return solution_found, solution_count, first_accuracy, time_to_first, time_to_correct, time_to_all


def search(
    name: str,
    query_generator: Callable,
    query_params: Dict[str, Any],
    context: Dict[str, Any],
    encoder: ImprovedSubgraphEncoder,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Full search pipeline: generate query -> embed -> FAISS search -> verify.
    
    Args:
        name: Experiment name for logging
        query_generator: Function to generate query
        query_params: Parameters for query generator
        context: Dict with graphs, indices, etc.
        encoder: Trained encoder model
        device: Torch device
        
    Returns:
        Dict with search results and metrics
    """
    # 1. Generate Query
    try:
        res = query_generator(**query_params)
        if res is None:
            return {"success": False, "error": "Query generation failed"}
        Gq, G_pos_vis, q_global_nodes, true_fine_indices, true_coarse_indices = res
    except Exception as e:
        return {"success": False, "error": str(e)}

    # 2. Embed Query
    zq = get_graph_embedding(Gq.to(device), encoder, device)

    # 3. Coarse Search (FAISS)
    D_coarse, I_coarse = context["faiss_coarse"].search(zq.cpu().numpy(), FAISS_TOP_K)
    predicted_coarse_idx = I_coarse[0][0].item()
    correct_coarse = predicted_coarse_idx in true_coarse_indices

    # 4. Fine Search within Coarse
    fine_graphs = context["fine_graphs"]
    fine_to_coarse = context["fine_to_coarse_map"]
    
    candidate_indices = [i for i, c in fine_to_coarse.items() if c == predicted_coarse_idx]
    if not candidate_indices:
        return {"success": False, "error": "No fine candidates in predicted coarse partition"}

    # Compute embeddings for candidates
    candidate_embeds = torch.cat([
        get_graph_embedding(fine_graphs[i].to(device), encoder, device) 
        for i in candidate_indices
    ], dim=0)
    
    faiss_fine = faiss.IndexFlatL2(candidate_embeds.shape[1])
    faiss_fine.add(candidate_embeds.cpu().numpy())
    _, I_fine = faiss_fine.search(zq.cpu().numpy(), 1)
    predicted_fine_global_idx = candidate_indices[I_fine[0][0].item()]
    correct_fine = predicted_fine_global_idx in true_fine_indices
    
    # 5. Glasgow Verification (if solver available)
    target_graph = context["coarse_graphs"][predicted_coarse_idx]
    target_nodes_map = context["coarse_part_nodes_map"][predicted_coarse_idx]
    target_local_to_global = {i: target_nodes_map[i].item() for i in range(len(target_nodes_map))}
    
    start_hint_gid = context["fine_part_nodes_map"][predicted_fine_global_idx][0].item()
    
    s_found, s_count, s_acc, t_first, t_corr, t_all = _run_glasgow_solver(
        Gq, q_global_nodes, target_graph, target_local_to_global, start_hint_gid, context
    )

    return {
        "success": True,
        "experiment_name": name,
        "coarse_correct": correct_coarse,
        "fine_correct": correct_fine,
        "solutions_found": s_count,
        "first_solution_accuracy": s_acc,
        "time_to_first_solution": t_first,
        "time_to_correct_solution": t_corr,
        "time_to_all_solutions": t_all,
        "perfect_solution_found": s_found and s_acc == 100,
        "predicted_coarse_idx": predicted_coarse_idx,
        "predicted_fine_idx": predicted_fine_global_idx,
        "true_fine_indices": true_fine_indices,
        "true_coarse_indices": list(true_coarse_indices),
    }
