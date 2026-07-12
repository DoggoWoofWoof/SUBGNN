import argparse
import csv
import random
import os
import networkx as nx
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import benchmark_retrieval as br
from src.data import load_dataset

def evaluate_random(queries, num_coarse):
    rows = []
    for query_number, item in enumerate(queries, start=1):
        true_coarse = item["true_coarse"]
        
        ranked_coarse = list(range(num_coarse))
        random.shuffle(ranked_coarse)
        
        for fixed_k in br.FIXED_KS:
            selected = ranked_coarse[:fixed_k]
            metrics = br.coverage_metrics(true_coarse, selected)
            rows.append({
                "method": "random",
                "query_id": item["query_id"],
                "target_query_size": item["target_query_size"],
                "seed_k": fixed_k,
                "seed_fullcov": metrics["fullcov"],
                "seed_recall": metrics["recall"],
            })
    return rows

def evaluate_pagerank(data, hierarchy, queries, num_coarse):
    rows = []
    print("Precomputing PageRank for coarse partitions...")
    node_to_coarse = hierarchy["node_to_coarse_map"]
    
    # Build coarse graph
    edges = data.edge_index
    src = edges[0]
    dst = edges[1]
    
    src_c = torch.tensor([node_to_coarse.get(n.item(), -1) for n in src])
    dst_c = torch.tensor([node_to_coarse.get(n.item(), -1) for n in dst])
    
    mask = (src_c >= 0) & (dst_c >= 0) & (src_c != dst_c)
    coarse_edges = torch.stack([src_c[mask], dst_c[mask]], dim=0)
    
    g_coarse = nx.Graph()
    g_coarse.add_nodes_from(range(num_coarse))
    g_coarse.add_edges_from(coarse_edges.t().tolist())
    
    pr = nx.pagerank(g_coarse)
    pr_scores = torch.tensor([pr.get(i, 0.0) for i in range(num_coarse)])
    ranked_coarse = torch.argsort(pr_scores, descending=True).tolist()
    
    print("Evaluating PageRank baseline...")
    for query_number, item in enumerate(tqdm(queries, desc="Query PR")):
        true_coarse = item["true_coarse"]
        for fixed_k in br.FIXED_KS:
            selected = ranked_coarse[:fixed_k]
            metrics = br.coverage_metrics(true_coarse, selected)
            rows.append({
                "method": "pagerank",
                "query_id": item["query_id"],
                "target_query_size": item["target_query_size"],
                "seed_k": fixed_k,
                "seed_fullcov": metrics["fullcov"],
                "seed_recall": metrics["recall"],
            })
    return rows


from torch_geometric.utils import subgraph

def compute_topofeat(graph_data, nodes):
    """Compute combined topological and feature signature for a set of nodes."""
    feats = graph_data.x[nodes].float().mean(dim=0)
    
    # Structural features
    sub_edge_index, _ = subgraph(nodes, graph_data.edge_index, relabel_nodes=True)
    if sub_edge_index.numel() > 0:
        g = nx.Graph()
        g.add_nodes_from(range(len(nodes)))
        g.add_edges_from(sub_edge_index.t().tolist())
        density = nx.density(g)
        avg_deg = sum(dict(g.degree()).values()) / max(1, len(nodes))
        max_deg = max(dict(g.degree()).values()) if len(nodes) > 0 else 0
    else:
        density, avg_deg, max_deg = 0.0, 0.0, 0.0
        
    topo_vec = torch.tensor([density, avg_deg, max_deg], dtype=torch.float)
    return torch.cat([feats, topo_vec])

def evaluate_topofeat(data, hierarchy, queries, num_coarse):
    rows = []
    print("Precomputing TopoFeat signatures for coarse partitions...")
    
    signatures = []
    node_to_coarse = hierarchy["node_to_coarse_map"]
    
    # Group nodes by partition
    part_nodes = {i: [] for i in range(num_coarse)}
    for node, part in node_to_coarse.items():
        if 0 <= int(part) < num_coarse:
            part_nodes[int(part)].append(int(node))
            
    for i in tqdm(range(num_coarse), desc="Partition Signatures"):
        nodes = torch.tensor(part_nodes[i], dtype=torch.long)
        if len(nodes) > 0:
            sig = compute_topofeat(data, nodes)
        else:
            sig = torch.zeros(data.x.size(1) + 3)
        signatures.append(sig)
        
    coarse_sigs = torch.stack(signatures)
    # L2 Normalize
    coarse_sigs = F.normalize(coarse_sigs, p=2, dim=1)
    
    print("Evaluating TopoFeat baseline...")
    for query_number, item in enumerate(tqdm(queries, desc="Query Signatures")):
        query = item["query"]
        true_coarse = item["true_coarse"]
        
        # The query graph is isolated, so its edge_index is internal
        # We can just use the query graph directly
        feats = query.x.float().mean(dim=0)
        if query.edge_index.numel() > 0:
            g = nx.Graph()
            g.add_nodes_from(range(query.num_nodes))
            g.add_edges_from(query.edge_index.t().tolist())
            density = nx.density(g)
            avg_deg = sum(dict(g.degree()).values()) / max(1, query.num_nodes)
            max_deg = max(dict(g.degree()).values()) if query.num_nodes > 0 else 0
        else:
            density, avg_deg, max_deg = 0.0, 0.0, 0.0
            
        topo_vec = torch.tensor([density, avg_deg, max_deg], dtype=torch.float)
        query_sig = torch.cat([feats, topo_vec]).unsqueeze(0)
        query_sig = F.normalize(query_sig, p=2, dim=1)
        
        # Cosine similarity
        sim = torch.mm(query_sig, coarse_sigs.t()).squeeze(0)
        
        # Rank descending
        ranked_coarse = torch.argsort(sim, descending=True).tolist()
        
        for fixed_k in br.FIXED_KS:
            selected = ranked_coarse[:fixed_k]
            metrics = br.coverage_metrics(true_coarse, selected)
            rows.append({
                "method": "topofeat",
                "query_id": item["query_id"],
                "target_query_size": item["target_query_size"],
                "seed_k": fixed_k,
                "seed_fullcov": metrics["fullcov"],
                "seed_recall": metrics["recall"],
            })
    return rows

def evaluate_mean_feature(data, hierarchy, queries, num_coarse):
    rows = []
    print("Precomputing mean features for coarse partitions...")
    node_to_coarse = hierarchy["node_to_coarse_map"]
    dim = data.x.size(1)
    
    coarse_sums = torch.zeros((num_coarse, dim), dtype=torch.float)
    coarse_counts = torch.zeros(num_coarse, dtype=torch.float)
    
    for node, part in node_to_coarse.items():
        if int(part) >= 0 and int(part) < num_coarse:
            coarse_sums[int(part)] += data.x[int(node)].float()
            coarse_counts[int(part)] += 1
        
    coarse_means = coarse_sums / coarse_counts.clamp(min=1.0).unsqueeze(1)
    coarse_means = F.normalize(coarse_means, p=2, dim=1)
    
    print("Evaluating MeanFeature baseline...")
    for query_number, item in enumerate(queries, start=1):
        query = item["query"]
        true_coarse = item["true_coarse"]
        
        query_mean = query.x.float().mean(dim=0, keepdim=True)
        query_mean = F.normalize(query_mean, p=2, dim=1)
        
        sim = torch.mm(query_mean, coarse_means.t()).squeeze(0)
        ranked_coarse = torch.argsort(sim, descending=True).tolist()
        
        for fixed_k in br.FIXED_KS:
            selected = ranked_coarse[:fixed_k]
            metrics = br.coverage_metrics(true_coarse, selected)
            rows.append({
                "method": "mean_feature",
                "query_id": item["query_id"],
                "target_query_size": item["target_query_size"],
                "seed_k": fixed_k,
                "seed_fullcov": metrics["fullcov"],
                "seed_recall": metrics["recall"],
            })
    return rows

def evaluate_neighbor_expansion(data, hierarchy, queries, num_coarse):
    rows = []
    print("Precomputing data for Neighbor Expansion...")
    node_to_coarse = hierarchy["node_to_coarse_map"]
    dim = data.x.size(1)
    
    coarse_sums = torch.zeros((num_coarse, dim), dtype=torch.float)
    coarse_counts = torch.zeros(num_coarse, dtype=torch.float)
    for node, part in node_to_coarse.items():
        if 0 <= int(part) < num_coarse:
            coarse_sums[int(part)] += data.x[int(node)].float()
            coarse_counts[int(part)] += 1
            
    coarse_means = coarse_sums / coarse_counts.clamp(min=1.0).unsqueeze(1)
    coarse_means = F.normalize(coarse_means, p=2, dim=1)
    
    # Build coarse graph
    edges = data.edge_index
    src_c = torch.tensor([node_to_coarse.get(n.item(), -1) for n in edges[0]])
    dst_c = torch.tensor([node_to_coarse.get(n.item(), -1) for n in edges[1]])
    mask = (src_c >= 0) & (dst_c >= 0) & (src_c != dst_c)
    coarse_edges = torch.stack([src_c[mask], dst_c[mask]], dim=0)
    
    g_coarse = nx.Graph()
    g_coarse.add_nodes_from(range(num_coarse))
    g_coarse.add_edges_from(coarse_edges.t().tolist())
    
    print("Evaluating Neighbor Expansion baseline...")
    for query_number, item in enumerate(tqdm(queries, desc="Query NE")):
        query = item["query"]
        true_coarse = item["true_coarse"]
        
        query_mean = query.x.float().mean(dim=0, keepdim=True)
        query_mean = F.normalize(query_mean, p=2, dim=1)
        sim = torch.mm(query_mean, coarse_means.t()).squeeze(0)
        
        # Start BFS from the best feature match
        best_seed = int(torch.argmax(sim).item())
        
        # BFS order
        visited = set([best_seed])
        order = [best_seed]
        queue = [best_seed]
        
        while queue:
            curr = queue.pop(0)
            for neighbor in g_coarse.neighbors(curr):
                if neighbor not in visited:
                    visited.add(neighbor)
                    order.append(neighbor)
                    queue.append(neighbor)
                    
        # If BFS didn't cover everything (disconnected components), append the rest
        if len(order) < num_coarse:
            unvisited = list(set(range(num_coarse)) - visited)
            unvisited.sort(key=lambda x: sim[x].item(), reverse=True)
            order.extend(unvisited)
            
        for fixed_k in br.FIXED_KS:
            selected = order[:fixed_k]
            metrics = br.coverage_metrics(true_coarse, selected)
            rows.append({
                "method": "neighbor_expansion",
                "query_id": item["query_id"],
                "target_query_size": item["target_query_size"],
                "seed_k": fixed_k,
                "seed_fullcov": metrics["fullcov"],
                "seed_recall": metrics["recall"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="arxiv")
    parser.add_argument("--queries-per-seed", type=int, default=100)
    parser.add_argument("--seeds", default="31415,42")
    parser.add_argument("--target-sizes", default="20,50,100")
    parser.add_argument("--output", default="results_non_neural.csv")
    parser.add_argument("--data-root", default="/cache")
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    data = load_dataset(args.dataset, root=args.data_root)
    
    hierarchy_path = br.default_hierarchy_path(args.dataset)
    if not os.path.exists(hierarchy_path):
        raise FileNotFoundError(f"Hierarchy not found at {hierarchy_path}")
        
    try:
        hierarchy = torch.load(hierarchy_path, map_location="cpu", weights_only=False)[0]
    except Exception as e:
        print(f"Failed to load: {e}")
        return
        
    hierarchy = br.prepare_hierarchy(data, hierarchy)
    num_coarse = len(hierarchy["coarse_part_nodes_map"])
    
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    target_sizes = [
        int(value) for value in args.target_sizes.split(",") if value.strip()
    ]
    queries = []
    for seed in seeds:
        for size in target_sizes:
            queries.extend(
                br.generate_queries(data, hierarchy, args.queries_per_seed, size, seed=seed)
            )
            
    print(f"Generated {len(queries)} queries.")
    
    rows_random = evaluate_random(queries, num_coarse)
    rows_mean = evaluate_mean_feature(data, hierarchy, queries, num_coarse)
    rows_topofeat = evaluate_topofeat(data, hierarchy, queries, num_coarse)
    rows_pagerank = evaluate_pagerank(data, hierarchy, queries, num_coarse)
    rows_ne = evaluate_neighbor_expansion(data, hierarchy, queries, num_coarse)
    
    all_rows = rows_random + rows_mean + rows_topofeat + rows_pagerank + rows_ne
    
    with open(args.output, "w", newline="") as f:
        fieldnames = ["method", "query_id", "target_query_size", "seed_k", "seed_fullcov", "seed_recall"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
            
    print("Summarizing Results:")
    for method in ["random", "mean_feature", "topofeat", "pagerank", "neighbor_expansion"]:
        for k in br.FIXED_KS:
            fc = [r["seed_fullcov"] for r in all_rows if r["method"] == method and r["seed_k"] == k]
            if fc:
                print(f"  {method} @ {k}: {sum(fc)/len(fc):.2%}")
                
if __name__ == "__main__":
    main()
