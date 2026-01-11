import hashlib
from typing import Dict, List, Any
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def feature_to_label(vector: np.ndarray) -> str:
    """
    Converts a node feature vector to a SHA256 hash string.
    This discretizes continuous features for the Glasgow Subgraph Solver.
    """
    # Sanity check: if vector is float and not binary, this might be unstable.
    # We assume features are effectively categorical or binary for hashing to make sense.
    # For floats, we might need rounding, but for now using strict equality via string repr of indices.
    
    # Check if sparse (binary) or dense
    if np.all(np.isin(vector, [0, 1])):
        indices = np.where(vector == 1)[0]
        index_str = "_".join(map(str, indices))
    else:
        # Fallback for dense float features (Arxiv/MAG embeddings)
        # Rounding to 4 decimals to avoid float precision jitter
        vector_rounded = np.round(vector, 4)
        index_str = ",".join(map(str, vector_rounded))
        
    hash_str = hashlib.sha256(index_str.encode("utf-8")).hexdigest()
    return hash_str

def convert_graph_to_csv(
    data: Data,
    filename: str,
    global_id_to_label_feature: Dict[int, str],
    local_to_global_map: Dict[int, int],
):
    """
    Exports a PyG Data object to the CSV format required by Glasgow Solver.
    """
    edge_index = data.edge_index.cpu().numpy()
    if edge_index.shape[1] == 0:
        # Handle empty graph case if needed
        pd.DataFrame(columns=[0, 1]).to_csv(filename, header=False, index=False, mode="w")
    else:
        edge_data = edge_index.T
        edges_df = pd.DataFrame(edge_data)
        edges_df.to_csv(filename, header=False, index=False, mode="w")

    node_label_list = []
    for i in range(data.num_nodes):
        global_id = local_to_global_map[i]
        label = global_id_to_label_feature.get(global_id, "unknown")
        # Format: ID, Label, Domain-Label (we use hash for domain label)
        node_label_list.append([i, "", label])
        
    node_labels_df = pd.DataFrame(node_label_list)
    node_labels_df.to_csv(filename, header=False, index=False, mode="a")

def are_partitions_neighbors(
    G_nx: nx.Graph, nodes_a: List[int], nodes_b: List[int]
) -> bool:
    """
    Checks if two partitions (node sets) are connected in the original graph.
    """
    node_set_b = set(nodes_b)
    for u in nodes_a:
        # Check neighbors in G_nx
        # Note: G_nx neighbors might be slow for huge graphs. 
        # For MAG, we might need a sparse tensor check (optimized in data.py).
        try:
            for v in G_nx.neighbors(u):
                if v in node_set_b:
                    return True
        except (KeyError, nx.NetworkXError):
            continue
    return False

def plot_unified_results(
    Gq, 
    G_truth_vis, 
    q_global_nodes, 
    G_predicted, 
    true_fine_indices, 
    predicted_fine_idx, 
    experiment_name, 
    save_path: str = None,
    **kwargs
):
    """
    Plots a unified visualization of the hierarchical graph search results (V1 style).

    Creates a 3-panel plot:
    1. The query graph.
    2. The ground truth graph(s), with the query nodes highlighted and drawn on top.
    3. The predicted graph, with the query nodes highlighted and drawn on top.
    
    Args:
        Gq: Query graph (PyG Data)
        G_truth_vis: Ground truth/stitched partition graph
        q_global_nodes: List of global node IDs in query
        G_predicted: Predicted partition graph
        true_fine_indices: Set/list of true fine partition indices
        predicted_fine_idx: Predicted fine partition index
        experiment_name: Name for title
        save_path: If provided, save figure instead of showing
        **kwargs: Must include 'fine_part_nodes_map'
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    is_correct = predicted_fine_idx in true_fine_indices if true_fine_indices else False
    fig.suptitle(f"Hierarchical Search Results: {experiment_name}", fontsize=16)

    # ===================================================================
    # Plot 1: Query Graph
    # ===================================================================
    if Gq is not None and Gq.num_nodes > 0:
        Gq_nx = to_networkx(Gq, to_undirected=True)
        pos_q = nx.spring_layout(Gq_nx, seed=42)
        axes[0].set_title(f"Query Graph (Gq)\n{Gq.num_nodes} nodes", fontsize=12)
        nx.draw(Gq_nx, pos_q, ax=axes[0],
                node_color='#ff0000', edgecolors='black',
                node_size=90, with_labels=False, linewidths=1.8)
    else:
        axes[0].set_title("Query Graph (Empty)")
        axes[0].text(0.5, 0.5, "No query graph", ha='center', va='center')

    # ===================================================================
    # Plot 2: Ground Truth
    # ===================================================================
    fine_part_nodes_map = kwargs.get('fine_part_nodes_map', {})
    
    if G_truth_vis is not None and G_truth_vis.num_nodes > 0:
        G_truth_nx = to_networkx(G_truth_vis, to_undirected=True)
        pos_truth = nx.spring_layout(G_truth_nx, seed=42)
        axes[1].set_title(f"Ground Truth Partition(s)\nFine Part(s): {sorted(true_fine_indices) if true_fine_indices else 'N/A'}", fontsize=12)

        # Build global-to-local mapping
        if true_fine_indices and len(true_fine_indices) == 1:
            idx = list(true_fine_indices)[0]
            if idx in fine_part_nodes_map:
                true_global_nodes = fine_part_nodes_map[idx]
                if hasattr(true_global_nodes, 'tolist'):
                    true_global_nodes = true_global_nodes.tolist()
                global_to_local_map = {node: i for i, node in enumerate(true_global_nodes)}
            else:
                global_to_local_map = {}
        elif true_fine_indices and len(true_fine_indices) > 1:
            all_global_nodes = []
            for idx in true_fine_indices:
                if idx in fine_part_nodes_map:
                    nodes = fine_part_nodes_map[idx]
                    if hasattr(nodes, 'tolist'):
                        nodes = nodes.tolist()
                    all_global_nodes.extend(nodes)
            all_global_nodes = sorted(set(all_global_nodes))
            global_to_local_map = {g: i for i, g in enumerate(all_global_nodes)}
        else:
            global_to_local_map = {}

        q_nodes_set = set(q_global_nodes if q_global_nodes else [])
        q_nodes_local = [global_to_local_map[g] for g in q_nodes_set if g in global_to_local_map]
        partition_nodes_local = [n for n in G_truth_nx.nodes() if n not in q_nodes_local]

        # Draw layers: Edges -> Partition Nodes -> Query Nodes
        nx.draw_networkx_edges(G_truth_nx, pos_truth, ax=axes[1], alpha=0.7)
        
        if true_fine_indices and len(true_fine_indices) > 1:
            # Multi-partition coloring
            node_to_part = {}
            for idx in true_fine_indices:
                if idx in fine_part_nodes_map:
                    nodes = fine_part_nodes_map[idx]
                    if hasattr(nodes, 'tolist'):
                        nodes = nodes.tolist()
                    for n in nodes:
                        node_to_part[n] = idx
            
            cmap = plt.get_cmap('Set2')
            part_color_map = {idx: cmap(i % cmap.N) for i, idx in enumerate(sorted(true_fine_indices))}
            local_to_global = {v: k for k, v in global_to_local_map.items()}
            
            partition_colors = []
            for local_idx in partition_nodes_local:
                g = local_to_global.get(local_idx)
                p = node_to_part.get(g)
                partition_colors.append(part_color_map.get(p, '#cccccc'))
            
            nx.draw_networkx_nodes(G_truth_nx, pos_truth, ax=axes[1],
                                   nodelist=partition_nodes_local,
                                   node_color=partition_colors,
                                   node_size=90)
        else:
            nx.draw_networkx_nodes(G_truth_nx, pos_truth, ax=axes[1],
                                   nodelist=partition_nodes_local,
                                   node_color='#66c2a5',
                                   node_size=90)

        # Query nodes on top
        nx.draw_networkx_nodes(G_truth_nx, pos_truth, ax=axes[1],
                               nodelist=q_nodes_local,
                               node_color='#ff0000', edgecolors='black',
                               node_size=90, linewidths=1.8)
    else:
        axes[1].set_title("Ground Truth (Empty)")
        axes[1].text(0.5, 0.5, "No ground truth", ha='center', va='center')

    # ===================================================================
    # Plot 3: Predicted Partition
    # ===================================================================
    if G_predicted is not None and G_predicted.num_nodes > 0:
        G_pred_nx = to_networkx(G_predicted, to_undirected=True)
        pos_pred = nx.spring_layout(G_pred_nx, seed=42)
        status = "(CORRECT)" if is_correct else "(INCORRECT)"
        axes[2].set_title(f"Predicted Partition (#{predicted_fine_idx})\n{status}", fontsize=12)

        if predicted_fine_idx in fine_part_nodes_map:
            pred_global = fine_part_nodes_map[predicted_fine_idx]
            if hasattr(pred_global, 'tolist'):
                pred_global = pred_global.tolist()
            g_to_l_pred = {n: i for i, n in enumerate(pred_global)}
        else:
            g_to_l_pred = {}

        q_in_pred = [g_to_l_pred[g] for g in (q_global_nodes or []) if g in g_to_l_pred]
        other_in_pred = [n for n in G_pred_nx.nodes() if n not in q_in_pred]
        base_color = '#228B22' if is_correct else '#FF6347'

        nx.draw_networkx_edges(G_pred_nx, pos_pred, ax=axes[2], alpha=0.7)
        nx.draw_networkx_nodes(G_pred_nx, pos_pred, ax=axes[2],
                               nodelist=other_in_pred,
                               node_color=base_color,
                               node_size=90)
        nx.draw_networkx_nodes(G_pred_nx, pos_pred, ax=axes[2],
                               nodelist=q_in_pred,
                               node_color='#ff0000', edgecolors='black',
                               node_size=90, linewidths=1.8)
    else:
        axes[2].set_title("Predicted (Empty)")
        axes[2].text(0.5, 0.5, "No prediction", ha='center', va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.savefig(f"{experiment_name}_results.png", dpi=150, bbox_inches='tight')
        plt.close()
