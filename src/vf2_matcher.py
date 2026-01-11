"""
VF2 Subgraph Isomorphism Matcher

Uses NetworkX's VF2 algorithm for subgraph isomorphism checking.
This is the Python fallback before moving to Glasgow solver.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


@dataclass
class MatchResult:
    """Result of a VF2 subgraph matching attempt (V1-compatible)."""
    found: bool                                   # Perfect solution found
    num_solutions: int                            # Total solutions enumerated
    first_mapping: Optional[Dict[int, int]]       # query_node -> target_node
    latency_seconds: float                        # Total time taken
    node_accuracy: float                          # % of nodes correctly mapped to ground truth
    
    # V1-compatible timing fields
    first_solution_accuracy: float = -1.0         # Accuracy of first found solution
    time_to_first_solution: float = -1.0          # Time to find ANY solution
    time_to_correct_solution: float = -1.0        # Time to find PERFECT solution
    time_to_all_solutions: float = -1.0           # Time to enumerate all
    

def pyg_to_networkx(data: Data, node_labels: Optional[Dict[int, Any]] = None) -> nx.Graph:
    """
    Convert PyG Data to NetworkX graph with optional node labels.
    
    Args:
        data: PyG Data object
        node_labels: Optional dict mapping node_id -> label for matching
        
    Returns:
        NetworkX graph with 'label' attributes if provided
    """
    G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    
    if node_labels is not None:
        for node in G.nodes():
            G.nodes[node]['label'] = node_labels.get(node, 0)
    
    return G


def vf2_subgraph_match(
    query: nx.Graph,
    target: nx.Graph,
    use_node_labels: bool = True,
    max_solutions: int = 10,
    timeout_seconds: float = 60.0
) -> MatchResult:
    """
    Perform VF2 subgraph isomorphism matching.
    
    Args:
        query: Query graph (smaller, to find in target)
        target: Target graph (larger, to search within)
        use_node_labels: Whether to enforce node label matching
        max_solutions: Maximum number of solutions to enumerate
        timeout_seconds: Timeout for matching
        
    Returns:
        MatchResult with match statistics
    """
    from networkx.algorithms.isomorphism import GraphMatcher
    
    # Node match function (if using labels)
    if use_node_labels and 'label' in nx.get_node_attributes(query, 'label'):
        def node_match(n1_attrs, n2_attrs):
            return n1_attrs.get('label') == n2_attrs.get('label')
    else:
        node_match = None
    
    # Create matcher
    GM = GraphMatcher(target, query, node_match=node_match)
    
    start_time = time.time()
    solutions = []
    time_to_first = -1.0
    
    try:
        # Enumerate solutions up to max_solutions
        for mapping in GM.subgraph_isomorphisms_iter():
            now = time.time()
            if now - start_time > timeout_seconds:
                break
            
            # Track time to first solution
            if time_to_first < 0:
                time_to_first = now - start_time
            
            solutions.append(mapping)
            if len(solutions) >= max_solutions:
                break
    except Exception as e:
        print(f"[VF2] Error during matching: {e}")
    
    latency = time.time() - start_time
    found = len(solutions) > 0
    first_mapping = solutions[0] if solutions else None
    
    return MatchResult(
        found=found,
        num_solutions=len(solutions),
        first_mapping=first_mapping,
        latency_seconds=latency,
        node_accuracy=0.0,  # Will be computed externally with ground truth
        first_solution_accuracy=-1.0,  # Computed in vf2_verify_subgraph
        time_to_first_solution=time_to_first,
        time_to_correct_solution=time_to_first if found else -1.0,  # For VF2, first=correct
        time_to_all_solutions=latency
    )


def compute_mapping_accuracy(
    mapping: Dict[int, int],
    query_global_ids: torch.Tensor,
    target_global_ids: torch.Tensor
) -> float:
    """
    Compute accuracy of a node mapping against ground truth.
    
    A correct mapping means: query_global_ids[q] == target_global_ids[mapping[q]]
    
    Args:
        mapping: Dict of query_local -> target_local
        query_global_ids: Global IDs for query nodes
        target_global_ids: Global IDs for target nodes
        
    Returns:
        Accuracy as float [0, 1]
    """
    if not mapping:
        return 0.0
    
    correct = 0
    total = len(mapping)
    
    for q_local, t_local in mapping.items():
        q_global = query_global_ids[q_local].item()
        t_global = target_global_ids[t_local].item()
        if q_global == t_global:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def vf2_verify_subgraph(
    query_data: Data,
    target_data: Data,
    query_global_ids: torch.Tensor,
    target_global_ids: torch.Tensor,
    use_node_labels: bool = False,
    timeout_seconds: float = 30.0
) -> MatchResult:
    """
    High-level function to verify if query is subgraph of target.
    
    Args:
        query_data: PyG Data for query graph
        target_data: PyG Data for target graph
        query_global_ids: Global node IDs for query
        target_global_ids: Global node IDs for target
        use_node_labels: Whether to use node features for matching
        timeout_seconds: Timeout for VF2
        
    Returns:
        MatchResult with accuracy computed against ground truth
    """
    # Convert to NetworkX
    query_nx = pyg_to_networkx(query_data)
    target_nx = pyg_to_networkx(target_data)
    
    # Run VF2
    result = vf2_subgraph_match(
        query_nx, target_nx,
        use_node_labels=use_node_labels,
        max_solutions=1,  # Just need first for accuracy
        timeout_seconds=timeout_seconds
    )
    
    # Compute accuracy if found
    if result.found and result.first_mapping:
        result.node_accuracy = compute_mapping_accuracy(
            result.first_mapping,
            query_global_ids,
            target_global_ids
        )
    
    return result


# --- BATCH VERIFICATION ---

def batch_vf2_verify(
    queries: List[Tuple[Data, torch.Tensor]],
    targets: List[Tuple[Data, torch.Tensor]],
    timeout_per_query: float = 10.0
) -> List[MatchResult]:
    """
    Verify multiple query-target pairs.
    
    Args:
        queries: List of (query_data, query_global_ids)
        targets: List of (target_data, target_global_ids)
        timeout_per_query: Timeout per individual match
        
    Returns:
        List of MatchResults
    """
    assert len(queries) == len(targets), "Must have same number of queries and targets"
    
    results = []
    for (q_data, q_gids), (t_data, t_gids) in zip(queries, targets):
        result = vf2_verify_subgraph(
            q_data, t_data, q_gids, t_gids,
            timeout_seconds=timeout_per_query
        )
        results.append(result)
    
    return results
