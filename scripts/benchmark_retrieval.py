"""
Retrieval-only evaluation for Jigsaw.

Evaluates fixed FAISS retrieval and top-20-seeded boundary expansion without
building or invoking Glasgow. The same generated queries are reused for every
model so FullCov and recall comparisons are directly comparable.
"""

import argparse
import csv
import gc
import hashlib
import os
import random
import statistics
import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

import benchmark_glasgow as bench
from retrieval_strategies import (
    fine_parent_ranking,
    hybrid_boundary_expand,
    multi_view_consensus_rankings,
    prefix_preserving_rerank,
    ranked_neighbor_stitch,
    reciprocal_rank_fusion,
)
from src.data import load_dataset
from src.model import get_graph_embedding


FIXED_KS = (20, 50, 100)
DYNAMIC_BUDGETS = (50, 75, 100)
MULTIVIEW_BUDGETS = (20, 50, 75, 100)
FINE_LEVELS = (25, 50, 75, 100, 150, 200, 250, 300, 400, 625, 750)
DEFAULT_HYBRID_MODEL_WEIGHTS = (0.25, 0.5, 0.75)
DEFAULT_TELEPORT_EVERY = (5, 10)
STITCH_SEED_KS = (5, 10, 20)
STITCH_POOL_KS = (100,)
STITCH_BUDGETS = (20, 50, 100)
PREFIX_KS = (1, 2, 5, 10)
SIGNATURE_PRUNING_STRATEGIES = (
    "type",
    "type_degree",
    "type_rel",
    "type_feat8",
    "type_rel_feat8",
    "type_feat16",
    "type_rel_feat16",
    "type_feat32",
    "type_rel_feat32",
)


def parse_model(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be label=path")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--model must be label=path")
    return label, path


def mean(values):
    return statistics.fmean(values) if values else 0.0


def default_hierarchy_path(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == "mag":
        return "/cache/mag_hierarchies_type_rel_2000_fine5_finecov_v1.pt"
    return f"/cache/{dataset_name}_hierarchies_finecov_v1.pt"


def load_named_data(dataset_name, data_root):
    data = load_dataset(dataset_name, root=data_root)
    print(
        f"Loaded {dataset_name}: {data.num_nodes:,} nodes, "
        f"{data.edge_index.size(1):,} directed edges",
        flush=True,
    )
    return data


def _weighted_partition_graph(data, node_to_part, num_parts):
    part_ids = torch.full((data.num_nodes,), -1, dtype=torch.long)
    for node_id, part_id in node_to_part.items():
        part_ids[int(node_id)] = int(part_id)

    src, dst = data.edge_index.detach().cpu().long()
    p_src = part_ids[src]
    p_dst = part_ids[dst]
    mask = (p_src != p_dst) & (p_src >= 0) & (p_dst >= 0)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_parts))
    if mask.any():
        lo = torch.minimum(p_src[mask], p_dst[mask])
        hi = torch.maximum(p_src[mask], p_dst[mask])
        edges, counts = torch.unique(
            torch.stack([lo, hi], dim=1), dim=0, return_counts=True
        )
        for (left, right), count in zip(edges.tolist(), counts.tolist()):
            graph.add_edge(int(left), int(right), weight=int(count))
    return graph


def prepare_hierarchy(data, hierarchy):
    hierarchy = dict(hierarchy)
    node_to_coarse = {
        int(node): int(part)
        for node, part in hierarchy["node_to_coarse_map"].items()
    }
    hierarchy["node_to_coarse_map"] = node_to_coarse

    node_to_fine = {}
    for fine_id, nodes in hierarchy["fine_part_nodes_map"].items():
        for node in nodes.detach().cpu().long().tolist():
            node_to_fine[int(node)] = int(fine_id)
    hierarchy["node_to_fine_map"] = node_to_fine

    hierarchy["coarse_part_graph"] = _weighted_partition_graph(
        data, node_to_coarse, len(hierarchy["coarse_part_nodes_map"])
    )
    hierarchy["fine_part_graph"] = _weighted_partition_graph(
        data, node_to_fine, len(hierarchy["fine_part_nodes_map"])
    )
    hierarchy["_fine_embedding_cache"] = {}
    return hierarchy


def build_coarse_overlap_index(data, hierarchy):
    """Precompute one-hop boundary memberships without changing partitions.

    For a selected coarse partition p, overlap candidate assembly includes nodes
    in p plus nodes in neighboring partitions that share an edge with p. This is
    a node-containment diagnostic for exact solving; hard partition FullCov is
    still reported separately.
    """
    if "coarse_part_node_sets" not in hierarchy:
        hierarchy["coarse_part_node_sets"] = {
            int(part_id): set(int(node) for node in nodes.detach().cpu().long().tolist())
            for part_id, nodes in hierarchy["coarse_part_nodes_map"].items()
        }

    if (
        "node_boundary_coarse_parts" in hierarchy
        and "coarse_overlap_node_sets" in hierarchy
    ):
        return hierarchy

    print("Building one-hop coarse overlap index...", flush=True)
    start = time.perf_counter()
    num_parts = len(hierarchy["coarse_part_nodes_map"])
    part_ids = torch.full((data.num_nodes,), -1, dtype=torch.long)
    for node_id, part_id in hierarchy["node_to_coarse_map"].items():
        part_ids[int(node_id)] = int(part_id)

    coarse_node_sets = hierarchy["coarse_part_node_sets"]

    node_boundary_parts = defaultdict(set)
    overlap_nodes_by_part = defaultdict(set)
    src, dst = data.edge_index.detach().cpu().long()
    p_src = part_ids[src]
    p_dst = part_ids[dst]
    mask = (p_src != p_dst) & (p_src >= 0) & (p_dst >= 0)
    cross_indices = torch.nonzero(mask, as_tuple=False).flatten()
    chunk_size = 1_000_000
    for start_idx in range(0, int(cross_indices.numel()), chunk_size):
        idx = cross_indices[start_idx : start_idx + chunk_size]
        for u, v, pu, pv in zip(
            src[idx].tolist(),
            dst[idx].tolist(),
            p_src[idx].tolist(),
            p_dst[idx].tolist(),
        ):
            u = int(u)
            v = int(v)
            pu = int(pu)
            pv = int(pv)
            node_boundary_parts[v].add(pu)
            node_boundary_parts[u].add(pv)
            overlap_nodes_by_part[pu].add(v)
            overlap_nodes_by_part[pv].add(u)

    hierarchy["node_boundary_coarse_parts"] = {
        int(node): frozenset(int(part) for part in parts)
        for node, parts in node_boundary_parts.items()
    }
    hierarchy["coarse_overlap_node_sets"] = {
        int(part): frozenset(int(node) for node in nodes)
        for part, nodes in overlap_nodes_by_part.items()
    }
    print(
        "Built one-hop overlap index: "
        f"{len(hierarchy['node_boundary_coarse_parts']):,} boundary nodes, "
        f"{sum(len(nodes) for nodes in hierarchy['coarse_overlap_node_sets'].values()):,} "
        f"partition-node overlap memberships across {num_parts:,} partitions "
        f"in {time.perf_counter() - start:.1f}s.",
        flush=True,
    )
    return hierarchy


def _node_type_tensor(data):
    if hasattr(data, "node_type") and data.node_type is not None:
        return data.node_type.detach().cpu().long()
    # Older cached MAG queries predate preservation of ``node_type`` during
    # subgraph extraction.  The query payload still carries the exact type
    # one-hot block in x, so recover it without consulting target/global IDs.
    if (
        getattr(data, "feature_schema", None) == "mag_type_rel_v1"
        and hasattr(data, "node_types")
        and data.x is not None
    ):
        type_start = 128
        type_width = len(data.node_types)
        type_end = type_start + type_width
        if type_width > 0 and int(data.x.size(1)) >= type_end:
            type_features = data.x.detach().cpu().float()[:, type_start:type_end]
            return torch.argmax(type_features, dim=1).long()
    return torch.zeros(data.num_nodes, dtype=torch.long)


def _degree_bucket_tensor(data):
    endpoints = data.edge_index.detach().cpu().long().reshape(-1)
    degree = torch.bincount(endpoints, minlength=data.num_nodes).float()
    return torch.floor(torch.log2(degree + 1.0)).long().clamp(max=31)


def _feature_bit_hash(features, width=8):
    width = min(width, features.size(1))
    bit_hash = torch.zeros(features.size(0), dtype=torch.long)
    for bit in range(width):
        bit_hash += (features[:, bit].detach().cpu() > 0).long() << bit
    return bit_hash


def _relation_degree_hash(data):
    if (
        not hasattr(data, "feature_schema")
        or data.feature_schema != "mag_type_rel_v1"
        or not hasattr(data, "node_types")
        or not hasattr(data, "edge_types")
    ):
        return torch.zeros(data.num_nodes, dtype=torch.long), 1

    rel_start = 128 + len(data.node_types)
    rel_width = min(2 * len(data.edge_types), max(0, data.x.size(1) - rel_start), 8)
    if rel_width <= 0:
        return torch.zeros(data.num_nodes, dtype=torch.long), 1

    rel = data.x.detach().cpu().float()[:, rel_start : rel_start + rel_width]
    quantized = torch.zeros_like(rel, dtype=torch.long)
    quantized[rel > 0.0] = 1
    quantized[rel > 0.10] = 2
    quantized[rel > 0.40] = 3

    rel_hash = torch.zeros(data.num_nodes, dtype=torch.long)
    base = 1
    for col in range(rel_width):
        rel_hash += quantized[:, col] * base
        base *= 4
    return rel_hash, base


def _build_node_signature_tokens(data):
    node_type = _node_type_tensor(data)
    degree_bucket = _degree_bucket_tensor(data)
    rel_hash, rel_base = _relation_degree_hash(data)
    features = data.x.detach().cpu().float()
    feature_hash8 = _feature_bit_hash(features, width=8)
    feature_base8 = 256
    feature_hash16 = _feature_bit_hash(features, width=16)
    feature_base16 = 65536
    feature_hash32 = _feature_bit_hash(features, width=32)
    feature_base32 = 2**32

    return {
        "type": node_type,
        "type_degree": node_type * 32 + degree_bucket,
        "type_rel": node_type * rel_base + rel_hash,
        "type_feat8": node_type * feature_base8 + feature_hash8,
        "type_rel_feat8": (node_type * rel_base + rel_hash)
        * feature_base8
        + feature_hash8,
        "type_feat16": node_type * feature_base16 + feature_hash16,
        "type_rel_feat16": (node_type * rel_base + rel_hash)
        * feature_base16
        + feature_hash16,
        "type_feat32": node_type * feature_base32 + feature_hash32,
        "type_rel_feat32": (node_type * rel_base + rel_hash)
        * feature_base32
        + feature_hash32,
    }


def _count_tokens_for_nodes(tokens, nodes):
    if not nodes:
        return Counter()
    node_tensor = torch.tensor(list(nodes), dtype=torch.long)
    values, counts = torch.unique(tokens[node_tensor], return_counts=True)
    return Counter({int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())})


def build_signature_pruning_index(data, hierarchy):
    """Precompute Bloom-style exact signature counts for overlap pruning.

    These are exact token counters rather than probabilistic Bloom filters so the
    diagnostic is interpretable. A Bloom/Roaring representation can replace the
    counters later once the best token family is chosen.
    """
    if "signature_pruning_index" in hierarchy:
        return hierarchy
    if "coarse_overlap_node_sets" not in hierarchy:
        hierarchy = build_coarse_overlap_index(data, hierarchy)

    print("Building signature-pruning indexes...", flush=True)
    start = time.perf_counter()
    tokens_by_name = _build_node_signature_tokens(data)
    num_parts = len(hierarchy["coarse_part_nodes_map"])
    part_node_sets = hierarchy.get("coarse_part_node_sets", {})
    overlap_node_sets = hierarchy.get("coarse_overlap_node_sets", {})
    pruning_index = {}
    for name in SIGNATURE_PRUNING_STRATEGIES:
        strategy_start = time.perf_counter()
        tokens = tokens_by_name[name]
        part_counts = {}
        overlap_counts = {}
        for part_id in range(num_parts):
            part_counts[part_id] = _count_tokens_for_nodes(
                tokens, part_node_sets.get(part_id, ())
            )
            overlap_counts[part_id] = _count_tokens_for_nodes(
                tokens, overlap_node_sets.get(part_id, ())
            )
        pruning_index[name] = {
            "tokens": tokens,
            "part_counts": part_counts,
            "overlap_counts": overlap_counts,
        }
        print(
            f"  Signature index {name}: built in "
            f"{time.perf_counter() - strategy_start:.1f}s",
            flush=True,
        )
    hierarchy["signature_pruning_index"] = pruning_index
    print(
        f"Built all signature-pruning indexes in {time.perf_counter() - start:.1f}s.",
        flush=True,
    )
    return hierarchy


def generate_queries(data, hierarchy, count, target_size, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    edge_values = data.edge_type.long() if hasattr(data, "edge_type") and data.edge_type is not None else None
    adj_t = SparseTensor.from_edge_index(
        data.edge_index, edge_attr=edge_values, sparse_sizes=(data.num_nodes, data.num_nodes)
    )
    queries = []
    attempts = 0
    while len(queries) < count and attempts < count * 20:
        attempts += 1
        generated = bench.generate_k_hop_query(data, adj_t, target_size=target_size)
        if generated is None:
            continue
        query, query_nodes, _, _ = generated
        true_coarse = bench.determine_true_coarse(
            query_nodes, hierarchy["node_to_coarse_map"]
        )
        true_fine = bench.determine_true_fine(
            query_nodes, hierarchy["node_to_fine_map"]
        )
        if not true_coarse or not true_fine:
            continue
        queries.append(
            {
                "query_id": f"k_hop_q{target_size}_{len(queries)}",
                "target_query_size": int(target_size),
                "query": query,
                "query_nodes": query_nodes.detach().cpu().long(),
                "true_coarse": set(int(x) for x in true_coarse),
                "true_fine": set(int(x) for x in true_fine),
            }
        )

    if len(queries) != count:
        raise RuntimeError(f"Generated only {len(queries)}/{count} k-hop queries")
    print(f"Generated {len(queries)} fixed-seed k-hop queries.", flush=True)
    return queries


def coverage_metrics(true_ids, selected_ids):
    true_set = set(int(x) for x in true_ids)
    selected_set = set(int(x) for x in selected_ids)
    missed = sorted(true_set - selected_set)
    covered = len(true_set) - len(missed)
    return {
        "count": len(selected_set),
        "recall": covered / len(true_set) if true_set else 0.0,
        "fullcov": bool(true_set) and not missed,
        "missed": str(missed),
    }


def overlap_node_metrics(item, selected_ids, hierarchy):
    metric_start = time.perf_counter()
    selected = set(int(x) for x in selected_ids)
    query_nodes = [int(node) for node in item["query_nodes"].tolist()]
    node_to_coarse = hierarchy["node_to_coarse_map"]
    boundary_parts = hierarchy.get("node_boundary_coarse_parts", {})
    missed = []
    for node in query_nodes:
        home_part = int(node_to_coarse.get(int(node), -1))
        if home_part in selected:
            continue
        if selected.intersection(boundary_parts.get(int(node), ())):
            continue
        missed.append(int(node))

    part_node_sets = hierarchy.get("coarse_part_node_sets", {})
    overlap_node_sets = hierarchy.get("coarse_overlap_node_sets", {})
    candidate_nodes_upper_bound = sum(
        len(part_node_sets.get(part_id, ()))
        + len(overlap_node_sets.get(part_id, ()))
        for part_id in selected
    )

    covered = len(query_nodes) - len(missed)
    signature_start = time.perf_counter()
    signature_metrics = signature_pruning_metrics(
        item["query"], selected, hierarchy, candidate_nodes_upper_bound
    )
    signature_time = time.perf_counter() - signature_start
    return {
        "overlap1_node_count": len(query_nodes),
        "overlap1_candidate_nodes": candidate_nodes_upper_bound,
        "overlap1_candidate_nodes_is_upper_bound": True,
        "overlap1_node_recall": covered / len(query_nodes) if query_nodes else 0.0,
        "overlap1_node_fullcov": bool(query_nodes) and not missed,
        "overlap1_missed_nodes": str(missed[:50]),
        "overlap1_missed_node_count": len(missed),
        "overlap1_metric_time_seconds": time.perf_counter() - metric_start,
        "signature_pruning_time_seconds": signature_time,
        **signature_metrics,
    }


def signature_pruning_metrics(query, selected, hierarchy, overlap_candidate_nodes):
    pruning_index = hierarchy.get("signature_pruning_index")
    if not pruning_index:
        return {}

    metrics = {}
    query_tokens_by_name = _build_node_signature_tokens(query)
    for name in SIGNATURE_PRUNING_STRATEGIES:
        index = pruning_index.get(name)
        if not index:
            continue
        query_tokens = set(
            int(x) for x in query_tokens_by_name[name].detach().cpu().tolist()
        )
        pruned_count = 0
        part_counts = index["part_counts"]
        overlap_counts = index["overlap_counts"]
        for part_id in selected:
            part_id = int(part_id)
            part_counter = part_counts.get(part_id, {})
            overlap_counter = overlap_counts.get(part_id, {})
            for token in query_tokens:
                pruned_count += int(part_counter.get(token, 0))
                pruned_count += int(overlap_counter.get(token, 0))

        fraction = (
            pruned_count / overlap_candidate_nodes
            if overlap_candidate_nodes > 0
            else 0.0
        )
        reduction = (
            overlap_candidate_nodes / pruned_count if pruned_count > 0 else 0.0
        )
        metrics[f"sig_{name}_candidate_nodes"] = pruned_count
        metrics[f"sig_{name}_candidate_fraction"] = fraction
        metrics[f"sig_{name}_reduction_vs_overlap"] = reduction
    return metrics


def prefix_precision_metrics(true_ids, ranked_ids):
    true_set = set(int(x) for x in true_ids)
    ranked_ids = [int(x) for x in ranked_ids]
    metrics = {}
    for prefix_k in PREFIX_KS:
        prefix = ranked_ids[:prefix_k]
        hits = len(set(prefix) & true_set)
        possible = min(prefix_k, len(true_set))
        metrics[f"prefix_hits_at_{prefix_k}"] = hits
        metrics[f"prefix_precision_at_{prefix_k}"] = (
            hits / prefix_k if prefix_k > 0 else 0.0
        )
        metrics[f"prefix_norm_precision_at_{prefix_k}"] = (
            hits / possible if possible > 0 else 0.0
        )
    return metrics


def true_rank_metrics(true_ids, ranked_ids):
    rank_by_id = {int(item): rank + 1 for rank, item in enumerate(ranked_ids)}
    ranks = {int(item): rank_by_id.get(int(item), -1) for item in sorted(true_ids)}
    found = [rank for rank in ranks.values() if rank > 0]
    return str(ranks), max(found) if found else -1


def build_coarse_mean_features(data, hierarchy):
    """Return one normalized mean-feature vector per coarse partition."""
    num_parts = len(hierarchy["coarse_part_nodes_map"])
    features = data.x.detach().cpu().float()
    sums = torch.zeros((num_parts, features.size(1)), dtype=torch.float32)
    counts = torch.zeros((num_parts,), dtype=torch.float32)
    for node_id, part_id in hierarchy["node_to_coarse_map"].items():
        part_id = int(part_id)
        if 0 <= part_id < num_parts:
            sums[part_id] += features[int(node_id)]
            counts[part_id] += 1.0
    means = sums / counts.clamp_min(1.0).unsqueeze(1)
    return F.normalize(means, p=2, dim=1)


def rank_by_mean_feature(query, coarse_mean_features):
    query_mean = query.x.detach().cpu().float().mean(dim=0, keepdim=True)
    query_mean = F.normalize(query_mean, p=2, dim=1)
    scores = torch.mm(query_mean, coarse_mean_features.t()).squeeze(0)
    return torch.argsort(scores, descending=True).tolist()


def build_connected_query_views(query, view_count=6, view_fraction=0.6):
    """Create deterministic connected query fragments without using labels or truth."""
    num_nodes = int(query.num_nodes)
    if num_nodes <= 4 or view_count <= 0:
        return []

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(query.edge_index.detach().cpu().long().T.tolist())
    view_size = min(num_nodes - 1, max(4, int(round(num_nodes * view_fraction))))

    # Farthest-first anchors spread views across the query while remaining
    # deterministic. Degree breaks distance ties toward informative centers.
    anchors = [
        min(graph.nodes, key=lambda node: (-graph.degree[node], int(node)))
    ]
    while len(anchors) < min(view_count, num_nodes):
        distance_maps = [
            nx.single_source_shortest_path_length(graph, anchor)
            for anchor in anchors
        ]

        def anchor_key(node):
            min_distance = min(
                distances.get(node, num_nodes + 1) for distances in distance_maps
            )
            return (-min_distance, -graph.degree[node], int(node))

        candidate = min(
            (node for node in graph.nodes if node not in anchors),
            key=anchor_key,
            default=None,
        )
        if candidate is None:
            break
        anchors.append(candidate)

    views = []
    seen = set()
    for anchor in anchors:
        ordered = []
        queued = {int(anchor)}
        queue = [int(anchor)]
        while queue and len(ordered) < view_size:
            node = queue.pop(0)
            ordered.append(node)
            for neighbor in sorted(int(value) for value in graph.neighbors(node)):
                if neighbor not in queued:
                    queued.add(neighbor)
                    queue.append(neighbor)
        node_ids = tuple(sorted(ordered))
        if len(node_ids) < 4 or node_ids in seen:
            continue
        seen.add(node_ids)
        local_ids = torch.tensor(
            node_ids, dtype=torch.long, device=query.edge_index.device
        )
        views.append(query.subgraph(local_ids))
    return views


def _rank_faiss_query(query, encoder, faiss_index, faiss_to_coarse, device):
    zq = get_graph_embedding(query.to(device), encoder, device)
    _, indices = faiss_index.search(
        zq.detach().cpu().numpy(), faiss_index.ntotal
    )
    return [
        int(faiss_to_coarse.get(int(index), int(index)))
        for index in indices[0]
        if int(index) >= 0
    ]


def evaluate_model_multiview(
    label,
    model_path,
    data,
    hierarchy,
    queries,
    device,
    view_count,
    view_fraction,
    support_depth,
):
    """Evaluate same-model connected-query-view fusion without other retrievers."""
    print(f"\nEvaluating multi-view retrieval for {label}: {model_path}", flush=True)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Required model is missing: {model_path}")
    encoder, model_load_time = bench.load_model(model_path, data.x.size(1), device)
    faiss_index, faiss_to_coarse, faiss_build_time = bench.build_faiss_index(
        data, hierarchy, encoder, device
    )

    rows = []
    for query_number, item in enumerate(queries, start=1):
        query = item["query"]
        true_coarse = item["true_coarse"]
        retrieval_start = time.perf_counter()
        views = build_connected_query_views(query, view_count, view_fraction)
        full_ranking = _rank_faiss_query(
            query, encoder, faiss_index, faiss_to_coarse, device
        )
        view_rankings = [
            _rank_faiss_query(view, encoder, faiss_index, faiss_to_coarse, device)
            for view in views
        ]
        retrieval_time = time.perf_counter() - retrieval_start
        rankings = {
            "fixed": full_ranking,
            **multi_view_consensus_rankings(
                full_ranking,
                view_rankings,
                support_depth=support_depth,
            ),
        }

        common = {
            "model": label,
            "model_path": model_path,
            "query_id": item["query_id"],
            "target_query_size": item["target_query_size"],
            "query_nodes": query.num_nodes,
            "query_views": len(views),
            "true_coarse_count": len(true_coarse),
            "true_fine_count": len(item["true_fine"]),
            "model_load_time": model_load_time,
            "faiss_build_time": faiss_build_time,
            "retrieval_time_seconds": retrieval_time,
        }
        for method, ranking in rankings.items():
            true_ranks, max_true_rank = true_rank_metrics(true_coarse, ranking)
            for budget in MULTIVIEW_BUDGETS:
                metrics = coverage_metrics(true_coarse, ranking[:budget])
                rows.append(
                    {
                        **common,
                        "true_coarse_ranks": true_ranks,
                        "max_true_coarse_rank": max_true_rank,
                        "method": method,
                        "seed_k": budget,
                        "coarse_budget": budget,
                        "seed_recall": metrics["recall"],
                        "seed_fullcov": metrics["fullcov"],
                        "seed_missed": metrics["missed"],
                        "expanded_count": metrics["count"],
                        "expanded_recall": metrics["recall"],
                        "expanded_fullcov": metrics["fullcov"],
                        "expanded_missed": metrics["missed"],
                        **_empty_fine_metrics(),
                    }
                )
        print(
            f"  Multi-view query {query_number}/{len(queries)} complete "
            f"({len(views)} views)",
            flush=True,
        )

    del encoder, faiss_index
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _build_global_fine_embeddings(hierarchy, data, encoder, device):
    fine_ids = sorted(int(fid) for fid in hierarchy["fine_part_nodes_map"])
    print(f"  Building global fine index for {len(fine_ids)} partitions", flush=True)
    fine_embeddings = torch.cat(
        [
            bench._get_fine_embedding(fid, hierarchy, data, encoder, device)
            for fid in fine_ids
        ],
        dim=0,
    )
    return fine_ids, fine_embeddings


def _rank_global_fine(zq, fine_ids, fine_embeddings):
    distances = torch.linalg.vector_norm(fine_embeddings - zq.detach().cpu(), dim=1)
    order = torch.argsort(distances).tolist()
    return [fine_ids[index] for index in order]


def _empty_fine_metrics():
    return {
        "fine_pool_count": 0,
        "fine_pool_recall": 0.0,
        "fine_pool_fullcov": False,
        "fine_boundary_min_fullcov_k": 0,
        **{f"fine_fullcov_at_{level}": False for level in FINE_LEVELS},
        **{f"fine_recall_at_{level}": 0.0 for level in FINE_LEVELS},
    }


def _fine_metrics(zq, expanded, true_fine, hierarchy, data, encoder, device):
    ranked_fine = bench._rank_fine_candidates(
        zq, expanded, hierarchy, data, encoder, device
    )
    fine_pool = coverage_metrics(true_fine, ranked_fine)
    fine_level_metrics = {}
    min_fullcov = 0
    for level in FINE_LEVELS:
        selected_fine = bench._select_fine_boundary_ids(
            ranked_fine,
            min(level, len(ranked_fine)),
            hierarchy["fine_part_graph"],
            seed_count=20,
        )
        metrics = coverage_metrics(true_fine, selected_fine)
        fine_level_metrics[level] = metrics
        if not min_fullcov and metrics["fullcov"]:
            min_fullcov = level
    return {
        "fine_pool_count": fine_pool["count"],
        "fine_pool_recall": fine_pool["recall"],
        "fine_pool_fullcov": fine_pool["fullcov"],
        "fine_boundary_min_fullcov_k": min_fullcov,
        **{
            f"fine_fullcov_at_{level}": fine_level_metrics[level]["fullcov"]
            for level in FINE_LEVELS
        },
        **{
            f"fine_recall_at_{level}": fine_level_metrics[level]["recall"]
            for level in FINE_LEVELS
        },
    }


def _append_overlap_diagnostic_row(
    rows,
    common,
    item,
    hierarchy,
    method,
    selected_ids,
    seed_k,
    coarse_budget,
):
    true_coarse = item["true_coarse"]
    hard_metrics = coverage_metrics(true_coarse, selected_ids)
    rows.append(
        {
            **common,
            "method": f"{method}_overlap1_nodes",
            "seed_k": seed_k,
            "coarse_budget": coarse_budget,
            "seed_recall": hard_metrics["recall"],
            "seed_fullcov": hard_metrics["fullcov"],
            "seed_missed": hard_metrics["missed"],
            "expanded_count": hard_metrics["count"],
            "expanded_recall": hard_metrics["recall"],
            "expanded_fullcov": hard_metrics["fullcov"],
            "expanded_missed": hard_metrics["missed"],
            **overlap_node_metrics(item, selected_ids, hierarchy),
            **_empty_fine_metrics(),
        }
    )


def _deterministic_random_partitions(all_coarse_ids, query_id, budget, label):
    if budget >= len(all_coarse_ids):
        return list(all_coarse_ids)
    digest = hashlib.sha256(f"{label}:{query_id}:{budget}".encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    rng = random.Random(seed)
    return rng.sample(list(all_coarse_ids), budget)


def _append_overlap_diagnostics(
    rows,
    common,
    item,
    hierarchy,
    ranked_coarse,
    fused_coarse,
    ranked_mean_feature,
    mean_fused_coarse,
    hybrid_model_weights,
    teleport_intervals,
):
    fixed_sources = (
        ("fixed", ranked_coarse),
        ("global_fine_rrf_fixed", fused_coarse),
        ("mean_feature_fixed", ranked_mean_feature),
        ("coarse_mean_rrf_fixed", mean_fused_coarse),
    )
    for method, ranking in fixed_sources:
        for fixed_k in FIXED_KS:
            _append_overlap_diagnostic_row(
                rows,
                common,
                item,
                hierarchy,
                method,
                ranking[:fixed_k],
                fixed_k,
                fixed_k,
            )

    for source_name, score_ranking in (
        ("coarse", ranked_coarse),
        ("global_fine_rrf", fused_coarse),
    ):
        seed_ids = score_ranking[:20]
        for model_weight in hybrid_model_weights:
            for teleport_every in teleport_intervals:
                if model_weight != 0.5 or teleport_every != 10:
                    continue
                method = (
                    f"{source_name}_hybrid_mw{model_weight:g}_"
                    f"teleport{teleport_every}"
                )
                for budget in DYNAMIC_BUDGETS:
                    expanded = hybrid_boundary_expand(
                        seed_ids,
                        score_ranking,
                        budget,
                        hierarchy["coarse_part_graph"],
                        seed_count=20,
                        model_weight=model_weight,
                        teleport_every=teleport_every,
                    )
                    _append_overlap_diagnostic_row(
                        rows,
                        common,
                        item,
                        hierarchy,
                        method,
                        expanded,
                        20,
                        budget,
                    )

    stitch_sources = (
        ("coarse_stitch", ranked_coarse, None),
        ("coarse_mean_stitch", ranked_coarse, ranked_mean_feature),
        ("coarse_mean_rrf_stitch", mean_fused_coarse, ranked_mean_feature),
    )
    for source_name, score_ranking, feature_ranking in stitch_sources:
        for seed_count in STITCH_SEED_KS:
            for budget in STITCH_BUDGETS:
                stitched = ranked_neighbor_stitch(
                    score_ranking,
                    budget,
                    hierarchy["coarse_part_graph"],
                    seed_count=seed_count,
                    pool_k=100,
                    feature_ranked_ids=feature_ranking,
                )
                _append_overlap_diagnostic_row(
                    rows,
                    common,
                    item,
                    hierarchy,
                    f"{source_name}_s{seed_count}_p100",
                    stitched,
                    seed_count,
                    budget,
                )


def evaluate_model(
    label,
    model_path,
    data,
    hierarchy,
    queries,
    device,
    hybrid_model_weights,
    teleport_intervals,
    collect_local_fine_metrics,
    include_k_sweep,
    include_overlap_node_coverage,
):
    print(f"\nEvaluating {label}: {model_path}", flush=True)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Required model is missing: {model_path}")
    hierarchy["_fine_embedding_cache"] = {}
    encoder, model_load_time = bench.load_model(model_path, data.x.size(1), device)
    faiss_index, faiss_to_coarse, faiss_build_time = bench.build_faiss_index(
        data, hierarchy, encoder, device
    )
    fine_ids, global_fine_embeddings = _build_global_fine_embeddings(
        hierarchy, data, encoder, device
    )
    coarse_mean_features = build_coarse_mean_features(data, hierarchy)

    rows = []
    ranking_traces = {}
    for query_number, item in enumerate(queries, start=1):
        query = item["query"]
        true_coarse = item["true_coarse"]
        true_fine = item["true_fine"]

        coarse_retrieval_start = time.perf_counter()
        zq = get_graph_embedding(query.to(device), encoder, device)
        # Dynamic retrieval needs the complete neural ranking so teleports can
        # recover candidates beyond the fixed top-100 baseline.
        search_k = faiss_index.ntotal
        _, indices = faiss_index.search(zq.detach().cpu().numpy(), search_k)
        ranked_coarse = [
            int(faiss_to_coarse.get(int(index), int(index)))
            for index in indices[0]
            if int(index) >= 0
        ]
        coarse_retrieval_time = time.perf_counter() - coarse_retrieval_start
        fused_retrieval_start = time.perf_counter()
        ranked_global_fine = _rank_global_fine(
            zq, fine_ids, global_fine_embeddings
        )
        ranked_fine_parents = fine_parent_ranking(
            ranked_global_fine, hierarchy["fine_to_coarse_map"]
        )
        fused_coarse = reciprocal_rank_fusion(
            [ranked_coarse, ranked_fine_parents]
        )
        ranked_mean_feature = rank_by_mean_feature(query, coarse_mean_features)
        mean_fused_coarse = reciprocal_rank_fusion(
            [ranked_coarse, ranked_mean_feature]
        )
        fused_retrieval_time = (
            coarse_retrieval_time + time.perf_counter() - fused_retrieval_start
        )
        ranking_traces[item["query_id"]] = {
            "coarse": ranked_coarse,
            "global_fine": fused_coarse,
            "mean_feature": ranked_mean_feature,
            "coarse_mean_rrf": mean_fused_coarse,
            "hybrid": {},
        }
        true_ranks, max_true_rank = true_rank_metrics(true_coarse, ranked_coarse)

        common = {
            "model": label,
            "model_path": model_path,
            "query_id": item["query_id"],
            "target_query_size": item["target_query_size"],
            "query_nodes": query.num_nodes,
            "true_coarse_count": len(true_coarse),
            "true_fine_count": len(true_fine),
            "true_coarse_ranks": true_ranks,
            "max_true_coarse_rank": max_true_rank,
            "model_load_time": model_load_time,
            "faiss_build_time": faiss_build_time,
            "retrieval_time_seconds": coarse_retrieval_time,
            "fused_retrieval_time_seconds": fused_retrieval_time,
        }

        fixed_sources = (
            ("fixed", ranked_coarse),
            ("global_fine_rrf_fixed", fused_coarse),
            ("mean_feature_fixed", ranked_mean_feature),
            ("coarse_mean_rrf_fixed", mean_fused_coarse),
        )
        for method, ranking in fixed_sources:
            for fixed_k in FIXED_KS:
                coarse = coverage_metrics(true_coarse, ranking[:fixed_k])
                rows.append(
                    {
                        **common,
                        "method": method,
                        "seed_k": fixed_k,
                        "coarse_budget": fixed_k,
                        "seed_recall": coarse["recall"],
                        "seed_fullcov": coarse["fullcov"],
                        "seed_missed": coarse["missed"],
                        "expanded_count": coarse["count"],
                        "expanded_recall": coarse["recall"],
                        "expanded_fullcov": coarse["fullcov"],
                        "expanded_missed": coarse["missed"],
                        **_empty_fine_metrics(),
                    }
                )

        if include_k_sweep:
            sweep_sources = (
                ("fixed_k_sweep", ranked_coarse),
                ("mean_feature_k_sweep", ranked_mean_feature),
                ("coarse_mean_rrf_k_sweep", mean_fused_coarse),
            )
            for method, ranking in sweep_sources:
                for fixed_k in range(1, min(100, len(ranking)) + 1):
                    coarse = coverage_metrics(true_coarse, ranking[:fixed_k])
                    rows.append(
                        {
                            **common,
                            "method": method,
                            "seed_k": fixed_k,
                            "coarse_budget": fixed_k,
                            "seed_recall": coarse["recall"],
                            "seed_fullcov": coarse["fullcov"],
                            "seed_missed": coarse["missed"],
                            "expanded_count": coarse["count"],
                            "expanded_recall": coarse["recall"],
                            "expanded_fullcov": coarse["fullcov"],
                            "expanded_missed": coarse["missed"],
                            **_empty_fine_metrics(),
                        }
                    )

        ranking_sources = (
            ("coarse", ranked_coarse),
            ("global_fine_rrf", fused_coarse),
        )
        for source_name, score_ranking in ranking_sources:
            seed_ids = score_ranking[:20]
            seed_metrics = coverage_metrics(true_coarse, seed_ids)
            for model_weight in hybrid_model_weights:
                for teleport_every in teleport_intervals:
                    method = (
                        f"{source_name}_hybrid_mw{model_weight:g}_"
                        f"teleport{teleport_every}"
                    )
                    for budget in DYNAMIC_BUDGETS:
                        expanded = hybrid_boundary_expand(
                            seed_ids,
                            score_ranking,
                            budget,
                            hierarchy["coarse_part_graph"],
                            seed_count=20,
                            model_weight=model_weight,
                            teleport_every=teleport_every,
                        )
                        expanded_metrics = coverage_metrics(true_coarse, expanded)
                        rows.append(
                            {
                                **common,
                                "method": method,
                                "seed_k": 20,
                                "coarse_budget": budget,
                                "seed_recall": seed_metrics["recall"],
                                "seed_fullcov": seed_metrics["fullcov"],
                                "seed_missed": seed_metrics["missed"],
                                "expanded_count": expanded_metrics["count"],
                                "expanded_recall": expanded_metrics["recall"],
                                "expanded_fullcov": expanded_metrics["fullcov"],
                                "expanded_missed": expanded_metrics["missed"],
                                **(
                                    _fine_metrics(
                                        zq,
                                        expanded,
                                        true_fine,
                                        hierarchy,
                                        data,
                                        encoder,
                                        device,
                                    )
                                    if collect_local_fine_metrics
                                    else _empty_fine_metrics()
                                ),
                            }
                        )

        stitch_sources = (
            ("coarse_stitch", ranked_coarse, None),
            ("global_fine_rrf_stitch", fused_coarse, None),
            ("mean_feature_stitch", ranked_mean_feature, None),
            ("coarse_mean_stitch", ranked_coarse, ranked_mean_feature),
            ("coarse_mean_rrf_stitch", mean_fused_coarse, ranked_mean_feature),
        )
        for source_name, score_ranking, feature_ranking in stitch_sources:
            for seed_count in STITCH_SEED_KS:
                seed_metrics = coverage_metrics(true_coarse, score_ranking[:seed_count])
                for pool_k in STITCH_POOL_KS:
                    for budget in STITCH_BUDGETS:
                        stitched = ranked_neighbor_stitch(
                            score_ranking,
                            budget,
                            hierarchy["coarse_part_graph"],
                            seed_count=seed_count,
                            pool_k=pool_k,
                            feature_ranked_ids=feature_ranking,
                        )
                        expanded_metrics = coverage_metrics(true_coarse, stitched)
                        rows.append(
                            {
                                **common,
                                "method": f"{source_name}_s{seed_count}_p{pool_k}",
                                "seed_k": seed_count,
                                "coarse_budget": budget,
                                "seed_recall": seed_metrics["recall"],
                                "seed_fullcov": seed_metrics["fullcov"],
                                "seed_missed": seed_metrics["missed"],
                                "expanded_count": expanded_metrics["count"],
                                "expanded_recall": expanded_metrics["recall"],
                                "expanded_fullcov": expanded_metrics["fullcov"],
                                "expanded_missed": expanded_metrics["missed"],
                                **_empty_fine_metrics(),
                            }
                        )

        prefix_sources = (
            ("fixed_prefix_rerank", ranked_coarse, None),
            ("global_fine_rrf_prefix_rerank", fused_coarse, None),
            ("mean_feature_prefix_rerank", ranked_mean_feature, None),
            ("coarse_mean_rrf_prefix_rerank", mean_fused_coarse, ranked_mean_feature),
        )
        for source_name, score_ranking, feature_ranking in prefix_sources:
            for budget in FIXED_KS:
                reranked = prefix_preserving_rerank(
                    score_ranking,
                    budget,
                    hierarchy["coarse_part_graph"],
                    feature_ranked_ids=feature_ranking,
                    seed_count=5,
                )
                metrics = coverage_metrics(true_coarse, reranked)
                rows.append(
                    {
                        **common,
                        "method": source_name,
                        "seed_k": budget,
                        "coarse_budget": budget,
                        "seed_recall": metrics["recall"],
                        "seed_fullcov": metrics["fullcov"],
                        "seed_missed": metrics["missed"],
                        "expanded_count": metrics["count"],
                        "expanded_recall": metrics["recall"],
                        "expanded_fullcov": metrics["fullcov"],
                        "expanded_missed": metrics["missed"],
                        **prefix_precision_metrics(true_coarse, reranked),
                        **_empty_fine_metrics(),
                    }
                )

        prefix_seed_sources = (
            ("coarse_prefix_seed", ranked_coarse, None),
            ("global_fine_rrf_prefix_seed", fused_coarse, None),
            ("coarse_mean_rrf_prefix_seed", mean_fused_coarse, ranked_mean_feature),
        )
        for source_name, score_ranking, feature_ranking in prefix_seed_sources:
            seed_order = prefix_preserving_rerank(
                score_ranking,
                20,
                hierarchy["coarse_part_graph"],
                feature_ranked_ids=feature_ranking,
                seed_count=5,
            )
            for prefix_k in PREFIX_KS:
                seed_ids = seed_order[:prefix_k]
                seed_metrics = coverage_metrics(true_coarse, seed_ids)
                for model_weight in hybrid_model_weights:
                    for teleport_every in teleport_intervals:
                        method = (
                            f"{source_name}_hybrid_s{prefix_k}_"
                            f"mw{model_weight:g}_teleport{teleport_every}"
                        )
                        for budget in DYNAMIC_BUDGETS:
                            expanded = hybrid_boundary_expand(
                                seed_ids,
                                score_ranking,
                                budget,
                                hierarchy["coarse_part_graph"],
                                seed_count=prefix_k,
                                model_weight=model_weight,
                                teleport_every=teleport_every,
                            )
                            expanded_metrics = coverage_metrics(true_coarse, expanded)
                            rows.append(
                                {
                                    **common,
                                    "method": method,
                                    "seed_k": prefix_k,
                                    "coarse_budget": budget,
                                    "seed_recall": seed_metrics["recall"],
                                    "seed_fullcov": seed_metrics["fullcov"],
                                    "seed_missed": seed_metrics["missed"],
                                    "expanded_count": expanded_metrics["count"],
                                    "expanded_recall": expanded_metrics["recall"],
                                    "expanded_fullcov": expanded_metrics["fullcov"],
                                    "expanded_missed": expanded_metrics["missed"],
                                    **prefix_precision_metrics(true_coarse, seed_ids),
                                    **_empty_fine_metrics(),
                                }
                            )

                for pool_k in STITCH_POOL_KS:
                    method = f"{source_name}_stitch_s{prefix_k}_p{pool_k}"
                    for budget in STITCH_BUDGETS:
                        stitched = ranked_neighbor_stitch(
                            score_ranking,
                            budget,
                            hierarchy["coarse_part_graph"],
                            seed_count=prefix_k,
                            pool_k=pool_k,
                            seed_ranked_ids=seed_ids,
                            feature_ranked_ids=feature_ranking,
                        )
                        expanded_metrics = coverage_metrics(true_coarse, stitched)
                        rows.append(
                            {
                                **common,
                                "method": method,
                                "seed_k": prefix_k,
                                "coarse_budget": budget,
                                "seed_recall": seed_metrics["recall"],
                                "seed_fullcov": seed_metrics["fullcov"],
                                "seed_missed": seed_metrics["missed"],
                                "expanded_count": expanded_metrics["count"],
                                "expanded_recall": expanded_metrics["recall"],
                                "expanded_fullcov": expanded_metrics["fullcov"],
                                "expanded_missed": expanded_metrics["missed"],
                                **prefix_precision_metrics(true_coarse, seed_ids),
                                **_empty_fine_metrics(),
                                }
                            )

        if include_overlap_node_coverage:
            _append_overlap_diagnostics(
                rows,
                common,
                item,
                hierarchy,
                ranked_coarse,
                fused_coarse,
                ranked_mean_feature,
                mean_fused_coarse,
                hybrid_model_weights,
                teleport_intervals,
            )

        # Fuse complete selection orders from the two complementary hybrid
        # retrievers. This is deterministic and stays within the same final
        # budget; unlike an oracle union, it does not inspect true partitions.
        for model_weight in hybrid_model_weights:
            for teleport_every in teleport_intervals:
                coarse_hybrid_order = hybrid_boundary_expand(
                    ranked_coarse[:20],
                    ranked_coarse,
                    faiss_index.ntotal,
                    hierarchy["coarse_part_graph"],
                    seed_count=20,
                    model_weight=model_weight,
                    teleport_every=teleport_every,
                )
                fine_hybrid_order = hybrid_boundary_expand(
                    fused_coarse[:20],
                    fused_coarse,
                    faiss_index.ntotal,
                    hierarchy["coarse_part_graph"],
                    seed_count=20,
                    model_weight=model_weight,
                    teleport_every=teleport_every,
                )
                trace_key = f"mw{model_weight:g}_teleport{teleport_every}"
                ranking_traces[item["query_id"]]["hybrid"][trace_key] = {
                    "coarse": coarse_hybrid_order,
                    "global_fine": fine_hybrid_order,
                }
                consensus_rankings = (
                    (
                        "dual_hybrid_rrf",
                        reciprocal_rank_fusion(
                            [coarse_hybrid_order, fine_hybrid_order]
                        ),
                    ),
                    (
                        "fixed_dual_hybrid_rrf",
                        reciprocal_rank_fusion(
                            [ranked_coarse, coarse_hybrid_order, fine_hybrid_order]
                        ),
                    ),
                )
                for consensus_name, consensus_ranking in consensus_rankings:
                    method = (
                        f"{consensus_name}_mw{model_weight:g}_"
                        f"teleport{teleport_every}"
                    )
                    seed_metrics = coverage_metrics(
                        true_coarse, consensus_ranking[:20]
                    )
                    for budget in DYNAMIC_BUDGETS:
                        selected = consensus_ranking[:budget]
                        expanded_metrics = coverage_metrics(true_coarse, selected)
                        rows.append(
                            {
                                **common,
                                "method": method,
                                "seed_k": 20,
                                "coarse_budget": budget,
                                "seed_recall": seed_metrics["recall"],
                                "seed_fullcov": seed_metrics["fullcov"],
                                "seed_missed": seed_metrics["missed"],
                                "expanded_count": expanded_metrics["count"],
                                "expanded_recall": expanded_metrics["recall"],
                                "expanded_fullcov": expanded_metrics["fullcov"],
                                "expanded_missed": expanded_metrics["missed"],
                                **_empty_fine_metrics(),
                            }
                        )

        print(f"  Query {query_number}/{len(queries)} complete", flush=True)

    del encoder, faiss_index, global_fine_embeddings, coarse_mean_features
    hierarchy["_fine_embedding_cache"] = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, ranking_traces


def evaluate_model_overlap_only(
    label,
    model_path,
    data,
    hierarchy,
    queries,
    device,
):
    print(f"\nEvaluating overlap-only diagnostics for {label}: {model_path}", flush=True)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Required model is missing: {model_path}")
    encoder, model_load_time = bench.load_model(model_path, data.x.size(1), device)
    faiss_index, faiss_to_coarse, faiss_build_time = bench.build_faiss_index(
        data, hierarchy, encoder, device
    )
    coarse_mean_features = build_coarse_mean_features(data, hierarchy)
    all_coarse_ids = sorted(int(part_id) for part_id in hierarchy["coarse_part_nodes_map"])

    rows = []
    for query_number, item in enumerate(queries, start=1):
        query = item["query"]
        coarse_retrieval_start = time.perf_counter()
        zq = get_graph_embedding(query.to(device), encoder, device)
        _, indices = faiss_index.search(
            zq.detach().cpu().numpy(), faiss_index.ntotal
        )
        ranked_coarse = [
            int(faiss_to_coarse.get(int(index), int(index)))
            for index in indices[0]
            if int(index) >= 0
        ]
        coarse_retrieval_time = time.perf_counter() - coarse_retrieval_start
        ranked_mean_feature = rank_by_mean_feature(query, coarse_mean_features)
        mean_fused_coarse = reciprocal_rank_fusion(
            [ranked_coarse, ranked_mean_feature]
        )
        true_ranks, max_true_rank = true_rank_metrics(
            item["true_coarse"], ranked_coarse
        )
        common = {
            "model": label,
            "model_path": model_path,
            "query_id": item["query_id"],
            "target_query_size": item["target_query_size"],
            "query_nodes": query.num_nodes,
            "true_coarse_count": len(item["true_coarse"]),
            "true_fine_count": len(item["true_fine"]),
            "true_coarse_ranks": true_ranks,
            "max_true_coarse_rank": max_true_rank,
            "model_load_time": model_load_time,
            "faiss_build_time": faiss_build_time,
            "retrieval_time_seconds": coarse_retrieval_time,
            "fused_retrieval_time_seconds": coarse_retrieval_time,
        }

        _append_overlap_diagnostic_row(
            rows,
            {
                **common,
                "retrieval_time_seconds": 0.0,
                "fused_retrieval_time_seconds": 0.0,
            },
            item,
            hierarchy,
            "filter_all_partitions",
            all_coarse_ids,
            len(all_coarse_ids),
            len(all_coarse_ids),
        )

        for fixed_k in FIXED_KS:
            random_ids = _deterministic_random_partitions(
                all_coarse_ids, item["query_id"], fixed_k, label
            )
            _append_overlap_diagnostic_row(
                rows,
                {
                    **common,
                    "retrieval_time_seconds": 0.0,
                    "fused_retrieval_time_seconds": 0.0,
                },
                item,
                hierarchy,
                "random_fixed",
                random_ids,
                fixed_k,
                fixed_k,
            )

        for method, ranking in (
            ("fixed", ranked_coarse),
            ("mean_feature_fixed", ranked_mean_feature),
            ("coarse_mean_rrf_fixed", mean_fused_coarse),
        ):
            for fixed_k in FIXED_KS:
                _append_overlap_diagnostic_row(
                    rows,
                    common,
                    item,
                    hierarchy,
                    method,
                    ranking[:fixed_k],
                    fixed_k,
                    fixed_k,
                )

        seed_ids = ranked_coarse[:20]
        for budget in DYNAMIC_BUDGETS:
            expanded = hybrid_boundary_expand(
                seed_ids,
                ranked_coarse,
                budget,
                hierarchy["coarse_part_graph"],
                seed_count=20,
                model_weight=0.5,
                teleport_every=10,
            )
            _append_overlap_diagnostic_row(
                rows,
                common,
                item,
                hierarchy,
                "coarse_hybrid_mw0.5_teleport10",
                expanded,
                20,
                budget,
            )

        for seed_count in STITCH_SEED_KS:
            for budget in STITCH_BUDGETS:
                stitched = ranked_neighbor_stitch(
                    mean_fused_coarse,
                    budget,
                    hierarchy["coarse_part_graph"],
                    seed_count=seed_count,
                    pool_k=100,
                    feature_ranked_ids=ranked_mean_feature,
                )
                _append_overlap_diagnostic_row(
                    rows,
                    common,
                    item,
                    hierarchy,
                    f"coarse_mean_rrf_stitch_s{seed_count}_p100",
                    stitched,
                    seed_count,
                    budget,
                )

        if query_number % 10 == 0 or query_number == len(queries):
            print(
                f"  Overlap-only query {query_number}/{len(queries)} complete",
                flush=True,
            )

    del encoder, faiss_index, coarse_mean_features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def evaluate_cross_model_ensemble(
    model_rankings,
    queries,
    hierarchy,
    hybrid_model_weights,
    teleport_intervals,
):
    """Fuse complementary model rankings without inspecting true partitions."""
    labels = sorted(model_rankings)
    label = "+".join(labels)
    rows = []

    for query_number, item in enumerate(queries, start=1):
        query_id = item["query_id"]
        true_coarse = item["true_coarse"]
        true_fine = item["true_fine"]
        traces = [model_rankings[model][query_id] for model in labels]

        cross_rankings = {
            "cross_model_coarse_rrf": reciprocal_rank_fusion(
                [trace["coarse"] for trace in traces]
            ),
            "cross_model_global_fine_rrf": reciprocal_rank_fusion(
                [trace["global_fine"] for trace in traces]
            ),
            "cross_model_all_rrf": reciprocal_rank_fusion(
                [
                    ranking
                    for trace in traces
                    for ranking in (trace["coarse"], trace["global_fine"])
                ]
            ),
        }

        common = {
            "model": label,
            "model_path": ";".join(labels),
            "query_id": query_id,
            "target_query_size": item["target_query_size"],
            "query_nodes": item["query"].num_nodes,
            "true_coarse_count": len(true_coarse),
            "true_fine_count": len(true_fine),
            "model_load_time": 0.0,
            "faiss_build_time": 0.0,
            "retrieval_time_seconds": 0.0,
            "fused_retrieval_time_seconds": 0.0,
        }

        for method, ranking in cross_rankings.items():
            true_ranks, max_true_rank = true_rank_metrics(true_coarse, ranking)
            method_common = {
                **common,
                "true_coarse_ranks": true_ranks,
                "max_true_coarse_rank": max_true_rank,
            }
            for fixed_k in FIXED_KS:
                metrics = coverage_metrics(true_coarse, ranking[:fixed_k])
                rows.append(
                    {
                        **method_common,
                        "method": f"{method}_fixed",
                        "seed_k": fixed_k,
                        "coarse_budget": fixed_k,
                        "seed_recall": metrics["recall"],
                        "seed_fullcov": metrics["fullcov"],
                        "seed_missed": metrics["missed"],
                        "expanded_count": metrics["count"],
                        "expanded_recall": metrics["recall"],
                        "expanded_fullcov": metrics["fullcov"],
                        "expanded_missed": metrics["missed"],
                        **_empty_fine_metrics(),
                    }
                )

            seed_metrics = coverage_metrics(true_coarse, ranking[:20])
            for model_weight in hybrid_model_weights:
                for teleport_every in teleport_intervals:
                    expanded_order = hybrid_boundary_expand(
                        ranking[:20],
                        ranking,
                        len(ranking),
                        hierarchy["coarse_part_graph"],
                        seed_count=20,
                        model_weight=model_weight,
                        teleport_every=teleport_every,
                    )
                    expanded_method = (
                        f"{method}_hybrid_mw{model_weight:g}_"
                        f"teleport{teleport_every}"
                    )
                    for budget in DYNAMIC_BUDGETS:
                        metrics = coverage_metrics(
                            true_coarse, expanded_order[:budget]
                        )
                        rows.append(
                            {
                                **method_common,
                                "method": expanded_method,
                                "seed_k": 20,
                                "coarse_budget": budget,
                                "seed_recall": seed_metrics["recall"],
                                "seed_fullcov": seed_metrics["fullcov"],
                                "seed_missed": seed_metrics["missed"],
                                "expanded_count": metrics["count"],
                                "expanded_recall": metrics["recall"],
                                "expanded_fullcov": metrics["fullcov"],
                                "expanded_missed": metrics["missed"],
                                **_empty_fine_metrics(),
                            }
                        )

        for model_weight in hybrid_model_weights:
            for teleport_every in teleport_intervals:
                trace_key = f"mw{model_weight:g}_teleport{teleport_every}"
                hybrid_sources = [
                    trace["hybrid"][trace_key][source]
                    for trace in traces
                    for source in ("coarse", "global_fine")
                ]
                ranking = reciprocal_rank_fusion(hybrid_sources)
                true_ranks, max_true_rank = true_rank_metrics(true_coarse, ranking)
                seed_metrics = coverage_metrics(true_coarse, ranking[:20])
                method = (
                    f"cross_model_all_hybrid_rrf_mw{model_weight:g}_"
                    f"teleport{teleport_every}"
                )
                for budget in DYNAMIC_BUDGETS:
                    metrics = coverage_metrics(true_coarse, ranking[:budget])
                    rows.append(
                        {
                            **common,
                            "true_coarse_ranks": true_ranks,
                            "max_true_coarse_rank": max_true_rank,
                            "method": method,
                            "seed_k": 20,
                            "coarse_budget": budget,
                            "seed_recall": seed_metrics["recall"],
                            "seed_fullcov": seed_metrics["fullcov"],
                            "seed_missed": seed_metrics["missed"],
                            "expanded_count": metrics["count"],
                            "expanded_recall": metrics["recall"],
                            "expanded_fullcov": metrics["fullcov"],
                            "expanded_missed": metrics["missed"],
                            **_empty_fine_metrics(),
                        }
                    )

        print(
            f"  Cross-model query {query_number}/{len(queries)} complete",
            flush=True,
        )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("target_query_size", row.get("query_nodes", 0)),
                row["model"],
                row["method"],
                row["seed_k"],
                row["coarse_budget"],
            )
        ].append(row)

    summaries = []

    def row_precision(row, recall_key, count_key):
        selected_count = float(row.get(count_key, 0) or 0)
        if selected_count <= 0:
            return 0.0
        return (
            float(row.get(recall_key, 0.0))
            * float(row.get("true_coarse_count", 0))
            / selected_count
        )

    for (target_query_size, model, method, seed_k, budget), group in sorted(grouped.items()):
        summary = {
            "target_query_size": target_query_size,
            "model": model,
            "method": method,
            "seed_k": seed_k,
            "coarse_budget": budget,
            "queries": len(group),
            "impossible_at_budget": sum(
                int(row["true_coarse_count"]) > int(budget) for row in group
            ),
            "avg_true_coarse_count": mean(
                [float(row["true_coarse_count"]) for row in group]
            ),
            "max_true_coarse_count": max(
                int(row["true_coarse_count"]) for row in group
            ),
            "avg_query_nodes": mean([float(row["query_nodes"]) for row in group]),
            "seed_fullcov": sum(bool(row["seed_fullcov"]) for row in group),
            "avg_seed_recall": mean([float(row["seed_recall"]) for row in group]),
            "avg_seed_precision": mean(
                [row_precision(row, "seed_recall", "seed_k") for row in group]
            ),
            "expanded_fullcov": sum(bool(row["expanded_fullcov"]) for row in group),
            "avg_expanded_recall": mean([float(row["expanded_recall"]) for row in group]),
            "avg_expanded_precision": mean(
                [
                    row_precision(row, "expanded_recall", "expanded_count")
                    for row in group
                ]
            ),
            "overlap1_node_fullcov": sum(
                bool(row.get("overlap1_node_fullcov", False)) for row in group
            ),
            "avg_overlap1_node_recall": mean(
                [
                    float(row.get("overlap1_node_recall", 0.0))
                    for row in group
                    if "overlap1_node_recall" in row
                ]
            ),
            "avg_overlap1_candidate_nodes": mean(
                [
                    float(row.get("overlap1_candidate_nodes", 0.0))
                    for row in group
                    if "overlap1_candidate_nodes" in row
                ]
            ),
            "fine_pool_fullcov": sum(bool(row["fine_pool_fullcov"]) for row in group),
            "avg_fine_pool_recall": mean([float(row["fine_pool_recall"]) for row in group]),
            "avg_max_true_coarse_rank": mean(
                [float(row["max_true_coarse_rank"]) for row in group]
            ),
            "avg_model_load_time_seconds": mean(
                [float(row.get("model_load_time", 0.0)) for row in group]
            ),
            "avg_faiss_build_time_seconds": mean(
                [float(row.get("faiss_build_time", 0.0)) for row in group]
            ),
            "avg_retrieval_time_seconds": mean(
                [float(row.get("retrieval_time_seconds", 0.0)) for row in group]
            ),
            "avg_fused_retrieval_time_seconds": mean(
                [
                    float(
                        row.get(
                            "fused_retrieval_time_seconds",
                            row.get("retrieval_time_seconds", 0.0),
                        )
                    )
                    for row in group
                ]
            ),
            "avg_overlap1_metric_time_seconds": mean(
                [
                    float(row.get("overlap1_metric_time_seconds", 0.0))
                    for row in group
                ]
            ),
            "avg_signature_pruning_time_seconds": mean(
                [
                    float(row.get("signature_pruning_time_seconds", 0.0))
                    for row in group
                ]
            ),
        }
        for signature_name in SIGNATURE_PRUNING_STRATEGIES:
            count_key = f"sig_{signature_name}_candidate_nodes"
            fraction_key = f"sig_{signature_name}_candidate_fraction"
            reduction_key = f"sig_{signature_name}_reduction_vs_overlap"
            summary[f"avg_{count_key}"] = mean(
                [
                    float(row[count_key])
                    for row in group
                    if count_key in row and row[count_key] != ""
                ]
            )
            summary[f"avg_{fraction_key}"] = mean(
                [
                    float(row[fraction_key])
                    for row in group
                    if fraction_key in row and row[fraction_key] != ""
                ]
            )
            summary[f"avg_{reduction_key}"] = mean(
                [
                    float(row[reduction_key])
                    for row in group
                    if reduction_key in row and row[reduction_key] != ""
                ]
            )
        for level in FINE_LEVELS:
            summary[f"fine_fullcov_at_{level}"] = sum(
                bool(row[f"fine_fullcov_at_{level}"]) for row in group
            )
            summary[f"avg_fine_recall_at_{level}"] = mean(
                [float(row[f"fine_recall_at_{level}"]) for row in group]
            )
        for prefix_k in PREFIX_KS:
            precision_values = [
                float(row[f"prefix_precision_at_{prefix_k}"])
                for row in group
                if f"prefix_precision_at_{prefix_k}" in row
                and row[f"prefix_precision_at_{prefix_k}"] != ""
            ]
            norm_precision_values = [
                float(row[f"prefix_norm_precision_at_{prefix_k}"])
                for row in group
                if f"prefix_norm_precision_at_{prefix_k}" in row
                and row[f"prefix_norm_precision_at_{prefix_k}"] != ""
            ]
            summary[f"avg_prefix_precision_at_{prefix_k}"] = mean(precision_values)
            summary[f"avg_prefix_norm_precision_at_{prefix_k}"] = mean(
                norm_precision_values
            )
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summaries):
    print(
        "\n| Query Size | Model | Method | Seed K | Budget | Impossible | Seed FullCov | Expanded FullCov | "
        "Overlap1 Node FullCov | Avg Overlap1 Recall | Avg Overlap1 Nodes | "
        "Avg Seed Recall | Avg Seed Prec | Avg Expanded Recall | Avg Expanded Prec | Fine Pool FullCov | Fine FullCov@100 |",
        flush=True,
    )
    print(
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        flush=True,
    )
    for row in summaries:
        print(
            f"| {row['target_query_size']} | {row['model']} | {row['method']} | {row['seed_k']} | "
            f"{row['coarse_budget']} | {row['impossible_at_budget']}/{row['queries']} | "
            f"{row['seed_fullcov']}/{row['queries']} | "
            f"{row['expanded_fullcov']}/{row['queries']} | "
            f"{row['overlap1_node_fullcov']}/{row['queries']} | "
            f"{row['avg_overlap1_node_recall']:.4f} | "
            f"{row['avg_overlap1_candidate_nodes']:.1f} | "
            f"{row['avg_seed_recall']:.4f} | {row['avg_seed_precision']:.4f} | "
            f"{row['avg_expanded_recall']:.4f} | {row['avg_expanded_precision']:.4f} | "
            f"{row['fine_pool_fullcov']}/{row['queries']} | "
            f"{row['fine_fullcov_at_100']}/{row['queries']} |",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Jigsaw retrieval-only benchmark")
    parser.add_argument("--model", action="append", required=True, type=parse_model)
    parser.add_argument("--dataset", default="arxiv")
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--target-size", type=int, default=20)
    parser.add_argument(
        "--target-sizes",
        default="",
        help="Comma-separated query node sizes to benchmark, e.g. 20,50,100.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="/data/datasets")
    parser.add_argument(
        "--hierarchy-path", default=""
    )
    parser.add_argument(
        "--output-prefix", default="/data/results/retrieval_arxiv_khop_q30_seed42"
    )
    parser.add_argument(
        "--hybrid-model-weight",
        action="append",
        type=float,
        dest="hybrid_model_weights",
        help="Repeat to sweep model-vs-boundary weights; defaults to 0.25/0.5/0.75.",
    )
    parser.add_argument(
        "--teleport-every",
        action="append",
        type=int,
        dest="teleport_intervals",
        help="Repeat to sweep neural teleport intervals; defaults to 5/10.",
    )
    parser.add_argument(
        "--skip-local-fine-metrics",
        action="store_true",
        help="Skip fine-boundary diagnostics while retaining global fine-first retrieval.",
    )
    parser.add_argument(
        "--skip-cross-model-ensemble",
        action="store_true",
        help="Do not evaluate deterministic cross-model RRF when multiple models are given.",
    )
    parser.add_argument(
        "--multi-view-only",
        action="store_true",
        help="Evaluate fixed and same-model connected-query-view retrieval only.",
    )
    parser.add_argument("--multi-view-count", type=int, default=6)
    parser.add_argument("--multi-view-fraction", type=float, default=0.6)
    parser.add_argument("--multi-view-support-depth", type=int, default=20)
    parser.add_argument(
        "--include-k-sweep",
        action="store_true",
        help="Also summarize fixed-rank precision/FullCov for K=1..100.",
    )
    parser.add_argument(
        "--include-overlap-node-coverage",
        action="store_true",
        help=(
            "Also report one-hop boundary-overlap node containment diagnostics "
            "without changing the selected partition count."
        ),
    )
    parser.add_argument(
        "--overlap-diagnostics-only",
        action="store_true",
        help=(
            "Run only the lean coarse/mean/stitch overlap diagnostics, skipping "
            "global fine retrieval and cross-model ensembling."
        ),
    )
    parser.add_argument(
        "--include-signature-pruning",
        action="store_true",
        help=(
            "With overlap diagnostics, estimate candidate-node reduction from "
            "Bloom-style node signature filters."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_named_data(args.dataset, args.data_root)
    hierarchy_path = args.hierarchy_path or default_hierarchy_path(args.dataset)
    print(f"Loading hierarchy: {hierarchy_path}", flush=True)
    if not os.path.exists(hierarchy_path):
        raise FileNotFoundError(f"Required hierarchy is missing: {hierarchy_path}")
    hierarchies = torch.load(hierarchy_path, map_location="cpu")
    hierarchy = prepare_hierarchy(data, hierarchies[0])
    del hierarchies
    if (
        args.include_overlap_node_coverage
        or args.overlap_diagnostics_only
        or args.include_signature_pruning
    ):
        hierarchy = build_coarse_overlap_index(data, hierarchy)
    if args.include_signature_pruning:
        hierarchy = build_signature_pruning_index(data, hierarchy)

    target_sizes = (
        [
            int(size.strip())
            for size in args.target_sizes.split(",")
            if size.strip()
        ]
        if args.target_sizes
        else [args.target_size]
    )
    queries = []
    for target_size in target_sizes:
        queries.extend(
            generate_queries(
                data, hierarchy, args.queries, target_size, args.seed + target_size
            )
        )

    rows = []
    hybrid_model_weights = (
        tuple(args.hybrid_model_weights)
        if args.hybrid_model_weights
        else DEFAULT_HYBRID_MODEL_WEIGHTS
    )
    teleport_intervals = (
        tuple(args.teleport_intervals)
        if args.teleport_intervals
        else DEFAULT_TELEPORT_EVERY
    )
    model_rankings = {}
    for label, path in args.model:
        if args.overlap_diagnostics_only:
            rows.extend(
                evaluate_model_overlap_only(
                    label,
                    path,
                    data,
                    hierarchy,
                    queries,
                    device,
                )
            )
            continue
        if args.multi_view_only:
            rows.extend(
                evaluate_model_multiview(
                    label,
                    path,
                    data,
                    hierarchy,
                    queries,
                    device,
                    args.multi_view_count,
                    args.multi_view_fraction,
                    args.multi_view_support_depth,
                )
            )
            continue
        model_rows, rankings = evaluate_model(
            label,
            path,
            data,
            hierarchy,
            queries,
            device,
            hybrid_model_weights,
            teleport_intervals,
            not args.skip_local_fine_metrics,
            args.include_k_sweep,
            args.include_overlap_node_coverage,
        )
        rows.extend(model_rows)
        model_rankings[label] = rankings

    if (
        not args.multi_view_only
        and not args.overlap_diagnostics_only
        and len(model_rankings) > 1
        and not args.skip_cross_model_ensemble
    ):
        print("\nEvaluating deterministic cross-model ensemble", flush=True)
        rows.extend(
            evaluate_cross_model_ensemble(
                model_rankings,
                queries,
                hierarchy,
                hybrid_model_weights,
                teleport_intervals,
            )
        )

    summaries = summarize(rows)
    per_query_path = f"{args.output_prefix}_per_query.csv"
    summary_path = f"{args.output_prefix}_summary.csv"
    write_csv(per_query_path, rows)
    write_csv(summary_path, summaries)
    print_summary(summaries)
    print(f"\nPer-query results: {per_query_path}", flush=True)
    print(f"Summary results: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
