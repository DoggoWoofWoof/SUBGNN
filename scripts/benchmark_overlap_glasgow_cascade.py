"""
Progressive overlap/signature Glasgow cascade benchmark.

This is a focused follow-up to retrieval-only diagnostics. It tests whether the
retrieved MAG/Arxiv candidate buckets are actually solver-viable:

    K=20 -> K=50 -> K=100

At each bucket it builds selected coarse partitions plus one-hop overlap nodes,
optionally prunes candidate nodes by an exact query-signature filter, and runs
Glasgow until the first solution or timeout.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import statistics
import threading
import time
from collections import defaultdict

import numpy as np
import torch
from torch_sparse import SparseTensor
from tqdm import tqdm

import benchmark_glasgow as bench
from benchmark_retrieval import (
    _build_node_signature_tokens,
    build_coarse_overlap_index,
    build_coarse_mean_features,
    default_hierarchy_path,
    load_named_data,
    prepare_hierarchy,
    rank_by_mean_feature,
)
from retrieval_strategies import hybrid_boundary_expand, reciprocal_rank_fusion
from src.data import build_single_hierarchy
from src.glasgow_solver import glasgow_solve
from src.model import get_graph_embedding
from src.utils import feature_to_label

try:
    import faiss
except ModuleNotFoundError:
    faiss = None


class NumpyIndexFlatL2:
    def __init__(self, dim):
        self.dim = int(dim)
        self.vectors = np.empty((0, self.dim), dtype=np.float32)

    @property
    def ntotal(self):
        return int(self.vectors.shape[0])

    def add(self, vectors):
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"Expected vectors with shape (*, {self.dim}), got {arr.shape}")
        self.vectors = np.concatenate([self.vectors, arr], axis=0)

    def search(self, queries, k):
        q = np.asarray(queries, dtype=np.float32)
        dists = ((q[:, None, :] - self.vectors[None, :, :]) ** 2).sum(axis=2)
        order = np.argsort(dists, axis=1)[:, :k]
        row = np.arange(q.shape[0])[:, None]
        return dists[row, order], order


def make_l2_index(dim):
    if faiss is not None:
        return faiss.IndexFlatL2(dim)
    return NumpyIndexFlatL2(dim)


def mean(values):
    return statistics.fmean(values) if values else 0.0


def clean_tag(text):
    return str(text).replace(",", "_").replace(" ", "").replace("/", "_").replace("\\", "_")


def parse_budgets(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def unique_ordered(values):
    seen = set()
    out = []
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def safe_cache_key(*parts):
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def path_fingerprint(path):
    if not path or not os.path.exists(path):
        return str(path)
    stat = os.stat(path)
    return f"{os.path.basename(path)}:{stat.st_size}"


def maybe_load_legacy_cache(cache_dir, cache_path, suffix):
    if not cache_dir or os.path.exists(cache_path):
        return None
    candidates = [
        os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir)
        if name.endswith(f"_{suffix}.pt")
    ]
    if len(candidates) > 1 and suffix in {"signature_tokens", "feature_label_tokens"}:
        payloads = []
        first = None
        all_identical = True
        for candidate in candidates:
            try:
                payload = torch.load(candidate, map_location="cpu")
            except Exception as exc:
                print(f"Could not inspect legacy {suffix} cache {candidate}: {exc}", flush=True)
                all_identical = False
                break
            if first is None:
                first = payload
            elif not (
                isinstance(first, torch.Tensor)
                and isinstance(payload, torch.Tensor)
                and first.shape == payload.shape
                and first.dtype == payload.dtype
                and torch.equal(first, payload)
            ):
                all_identical = False
                break
            payloads.append(payload)
        if all_identical and first is not None:
            print(
                f"Loading legacy {suffix} cache by identical candidates: "
                f"{len(candidates)} files",
                flush=True,
            )
            torch.save(first, cache_path)
            print(f"Saved stable cache alias: {cache_path}", flush=True)
            return first
    if len(candidates) != 1:
        if candidates:
            print(
                f"Legacy cache fallback skipped for {suffix}: "
                f"{len(candidates)} candidates",
                flush=True,
            )
        return None
    print(
        f"Loading legacy {suffix} cache by suffix: {candidates[0]}",
        flush=True,
    )
    payload = torch.load(candidates[0], map_location="cpu")
    torch.save(payload, cache_path)
    print(f"Saved stable cache alias: {cache_path}", flush=True)
    return payload


def _first_valid_coarse_embedding(data, hierarchy, encoder, device):
    for coarse_id, graph in enumerate(hierarchy["coarse_graphs"]):
        if graph is None:
            continue
        graph = graph.to(device)
        if graph.x is None:
            gids = graph.global_id.to(data.x.device)
            graph.x = data.x[gids].to(device)
        emb = get_graph_embedding(graph, encoder, device).detach().cpu().float().view(-1)
        return int(coarse_id), emb
    return None, None


def maybe_load_legacy_coarse_embeddings(
    data, hierarchy, encoder, device, cache_dir, cache_path
):
    if not cache_dir or os.path.exists(cache_path):
        return None
    candidates = [
        os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir)
        if name.endswith("_coarse_embeddings.pt")
    ]
    if not candidates:
        return None
    coarse_id, reference = _first_valid_coarse_embedding(data, hierarchy, encoder, device)
    if reference is None:
        return None

    best = None
    for candidate in candidates:
        try:
            payload = torch.load(candidate, map_location="cpu")
            embeddings = payload["embeddings"].float()
            faiss_to_coarse = {
                int(k): int(v) for k, v in payload["faiss_to_coarse"].items()
            }
            row_idx = next(
                (idx for idx, cid in faiss_to_coarse.items() if cid == coarse_id),
                None,
            )
            if row_idx is None or int(row_idx) >= int(embeddings.size(0)):
                continue
            dist = torch.norm(embeddings[int(row_idx)].view(-1) - reference).item()
            if best is None or dist < best[0]:
                best = (dist, candidate, payload)
        except Exception as exc:
            print(f"Could not inspect legacy coarse embedding cache {candidate}: {exc}", flush=True)

    if best is None:
        return None
    dist, candidate, payload = best
    if dist > 1e-3:
        print(
            f"Legacy coarse embedding fallback skipped: best distance {dist:.6g} "
            f"from {candidate}",
            flush=True,
        )
        return None
    print(
        f"Loading legacy coarse embedding cache: {candidate} "
        f"(matched coarse {coarse_id}, distance {dist:.6g})",
        flush=True,
    )
    torch.save(payload, cache_path)
    print(f"Saved stable coarse embedding cache alias: {cache_path}", flush=True)
    return payload


def default_partition_config(dataset_name):
    configs = {
        "cora": (20, 5),
        "corafull": (20, 5),
        "citeseer": (10, 5),
        "pubmed": (20, 5),
        "physics": (35, 5),
        "flickr": (100, 5),
        "arxiv": (200, 5),
        "mag": (2000, 5),
        "yelp": (700, 5),
    }
    return configs.get(dataset_name.lower(), (20, 5))


def load_or_build_overlap_index(data, hierarchy, cache_dir, cache_key):
    if not cache_dir:
        return build_coarse_overlap_index(data, hierarchy)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{cache_key}_overlap_index.pt")
    if os.path.exists(path):
        print(f"Loading cached overlap index: {path}", flush=True)
        cached = torch.load(path, map_location="cpu")
        hierarchy.update(cached)
        return hierarchy
    cached = maybe_load_legacy_cache(cache_dir, path, "overlap_index")
    if cached is not None:
        hierarchy.update(cached)
        return hierarchy
    hierarchy = build_coarse_overlap_index(data, hierarchy)
    torch.save(
        {
            "node_boundary_coarse_parts": hierarchy["node_boundary_coarse_parts"],
            "coarse_part_node_sets": hierarchy["coarse_part_node_sets"],
            "coarse_overlap_node_sets": hierarchy["coarse_overlap_node_sets"],
        },
        path,
    )
    print(f"Saved overlap index cache: {path}", flush=True)
    return hierarchy


def load_or_prepare_hierarchy(data, hierarchy_path, cache_dir, cache_key, dataset_name=""):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{cache_key}_prepared_hierarchy.pt")
        if os.path.exists(path):
            print(f"Loading cached prepared hierarchy: {path}", flush=True)
            return torch.load(path, map_location="cpu")
        cached = maybe_load_legacy_cache(cache_dir, path, "prepared_hierarchy")
        if cached is not None:
            return cached
    if os.path.exists(hierarchy_path):
        print(f"Loading hierarchy: {hierarchy_path}", flush=True)
        hierarchies = torch.load(hierarchy_path, map_location="cpu")
        raw_hierarchy = hierarchies[0]
        del hierarchies
    else:
        num_coarse, num_fine = default_partition_config(
            dataset_name or args_dataset_from_path(hierarchy_path)
        )
        print(
            f"Hierarchy missing at {hierarchy_path}; building {num_coarse}x{num_fine}.",
            flush=True,
        )
        raw_hierarchy = build_single_hierarchy(data, num_coarse, num_fine)
        try:
            os.makedirs(os.path.dirname(hierarchy_path), exist_ok=True)
            torch.save([raw_hierarchy], hierarchy_path)
            print(f"Saved rebuilt hierarchy: {hierarchy_path}", flush=True)
        except Exception as exc:
            print(f"Could not save rebuilt hierarchy: {exc}", flush=True)
    hierarchy = prepare_hierarchy(data, raw_hierarchy)
    if cache_dir:
        torch.save(hierarchy, path)
        print(f"Saved prepared hierarchy cache: {path}", flush=True)
    return hierarchy


def args_dataset_from_path(path):
    base = os.path.basename(path).lower()
    for name in ("corafull", "citeseer", "pubmed", "physics", "flickr", "arxiv", "mag", "yelp", "cora"):
        if base.startswith(f"{name}_") or base == name:
            return name
    return "cora"


def load_or_build_signature_tokens(data, cache_dir, cache_key):
    if not cache_dir:
        return _build_node_signature_tokens(data)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{cache_key}_signature_tokens.pt")
    if os.path.exists(path):
        print(f"Loading cached signature tokens: {path}", flush=True)
        return torch.load(path, map_location="cpu")
    cached = maybe_load_legacy_cache(cache_dir, path, "signature_tokens")
    if cached is not None:
        return cached
    tokens = _build_node_signature_tokens(data)
    torch.save(tokens, path)
    print(f"Saved signature token cache: {path}", flush=True)
    return tokens


def load_or_build_feature_label_tokens(data, cache_dir, cache_key):
    if not cache_dir:
        return torch.tensor(
            [feature_to_label(data.x[i]) for i in range(data.num_nodes)],
            dtype=torch.long,
        )
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{cache_key}_feature_label_tokens.pt")
    if os.path.exists(path):
        print(f"Loading cached exact feature labels: {path}", flush=True)
        return torch.load(path, map_location="cpu").long()
    cached = maybe_load_legacy_cache(cache_dir, path, "feature_label_tokens")
    if cached is not None:
        return cached.long()
    print("Building exact feature-label cache...", flush=True)
    labels = torch.tensor(
        [feature_to_label(data.x[i]) for i in range(data.num_nodes)],
        dtype=torch.long,
    )
    torch.save(labels, path)
    print(f"Saved exact feature-label cache: {path}", flush=True)
    return labels


def _feature_bucket_hash_int(vector):
    if vector is None:
        return 0
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().cpu().numpy()
    if np.all(np.isin(vector, [0, 1])):
        feats_tuple = tuple(np.where(vector == 1)[0].tolist())
    else:
        feats_tuple = tuple(np.round(vector, 4).tolist())
    return int(hashlib.md5(str(feats_tuple).encode("utf-8")).hexdigest(), 16)


def load_or_build_feature_bucket_label_tokens(data, cache_dir, cache_key, bucket_count):
    if bucket_count <= 0:
        raise ValueError(f"feature bucket count must be positive, got {bucket_count}")
    if not cache_dir:
        return torch.tensor(
            [_feature_bucket_hash_int(data.x[i]) % bucket_count for i in range(data.num_nodes)],
            dtype=torch.long,
        )
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{cache_key}_feature_bucket_{bucket_count}_label_tokens.pt")
    if os.path.exists(path):
        print(f"Loading cached feature bucket labels K={bucket_count}: {path}", flush=True)
        return torch.load(path, map_location="cpu").long()
    print(f"Building feature bucket label cache K={bucket_count}...", flush=True)
    labels = torch.tensor(
        [_feature_bucket_hash_int(data.x[i]) % bucket_count for i in range(data.num_nodes)],
        dtype=torch.long,
    )
    torch.save(labels, path)
    print(f"Saved feature bucket label cache K={bucket_count}: {path}", flush=True)
    return labels


def derive_query_labels(query, label_source="feature", class_venue_base=None):
    """Derive solver labels solely from the query payload.

    Planted global node IDs are evaluation metadata and must never determine
    serving-time candidate labels.  Feature and bucket labels are deterministic
    from query.x.  Class-labelled queries must carry y or node_label, just as an
    external query request would.
    """
    label_source = str(label_source or "feature")
    if label_source == "feature":
        if query.x is None:
            raise ValueError("feature labels require query.x")
        return [feature_to_label(query.x[i]) for i in range(query.num_nodes)]
    if label_source.startswith("feature_bucket_"):
        if query.x is None:
            raise ValueError("feature-bucket labels require query.x")
        bucket_count = int(label_source.rsplit("_", 1)[-1])
        if bucket_count <= 0:
            raise ValueError(f"feature bucket count must be positive, got {bucket_count}")
        return [
            _feature_bucket_hash_int(query.x[i]) % bucket_count
            for i in range(query.num_nodes)
        ]
    if label_source == "class":
        explicit = getattr(query, "node_label", None)
        if isinstance(explicit, torch.Tensor) and int(explicit.numel()) == int(query.num_nodes):
            return [int(value) for value in explicit.detach().cpu().view(-1).tolist()]
        y = getattr(query, "y", None)
        if not isinstance(y, torch.Tensor) or int(y.numel()) != int(query.num_nodes):
            raise ValueError(
                "class labels require query.y or query.node_label; regenerate legacy "
                "query caches so labels are stored in the query payload"
            )
        if class_venue_base is None:
            nonnegative = y.detach().cpu().view(-1).long()
            class_venue_base = (
                int(nonnegative[nonnegative >= 0].max()) + 1
                if bool((nonnegative >= 0).any())
                else 1
            )
        from benchmark_retrieval import _node_type_tensor

        node_type = _node_type_tensor(query)
        y = y.detach().cpu().view(-1).long()
        within_type = torch.where(y >= 0, y, torch.zeros_like(y))
        values = node_type * int(class_venue_base) + within_type
        return [int(value) for value in values.tolist()]
    raise ValueError(f"Unknown label source: {label_source}")


def derive_query_signature_tokens(query, signature_name):
    """Build the active signature from query attributes, never target IDs."""
    if not signature_name or signature_name == "none":
        return torch.zeros(0, dtype=torch.long)
    query_tokens_by_name = _build_node_signature_tokens(query)
    if signature_name not in query_tokens_by_name:
        raise ValueError(
            f"Unknown signature {signature_name}; options: {sorted(query_tokens_by_name)}"
        )
    return torch.unique(query_tokens_by_name[signature_name].detach().cpu().long())


def build_or_load_faiss_index(data, hierarchy, encoder, device, model_path, cache_dir, cache_key):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    model_key = safe_cache_key(cache_key, path_fingerprint(model_path), data.x.size(1))
    cache_path = os.path.join(cache_dir, f"{model_key}_coarse_embeddings.pt") if cache_dir else ""
    payload = None
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached coarse embeddings: {cache_path}", flush=True)
        payload = torch.load(cache_path, map_location="cpu")
    elif cache_path:
        payload = maybe_load_legacy_coarse_embeddings(
            data, hierarchy, encoder, device, cache_dir, cache_path
        )
    if payload is not None:
        coarse_embeds = payload["embeddings"].float()
        faiss_to_coarse = {int(k): int(v) for k, v in payload["faiss_to_coarse"].items()}
        start = time.perf_counter()
        index = make_l2_index(coarse_embeds.size(1))
        index.add(coarse_embeds.numpy())
        print(f"  FAISS index from cache: {index.ntotal} vectors ({time.perf_counter() - start:.1f}s)", flush=True)
        return index, faiss_to_coarse, 0.0

    start = time.perf_counter()
    embeds = []
    faiss_to_coarse = {}
    for coarse_id, graph in enumerate(tqdm(hierarchy["coarse_graphs"], desc="  Building FAISS index", ncols=80)):
        if graph is None:
            continue
        graph = graph.to(device)
        if graph.x is None:
            gids = graph.global_id.to(data.x.device)
            graph.x = data.x[gids].to(device)
        emb = get_graph_embedding(graph, encoder, device).detach().cpu()
        faiss_to_coarse[len(embeds)] = int(coarse_id)
        embeds.append(emb)
    coarse_embeds = torch.cat(embeds, dim=0).float()
    index = make_l2_index(coarse_embeds.size(1))
    index.add(coarse_embeds.numpy())
    build_time = time.perf_counter() - start
    print(f"  FAISS index: {index.ntotal} vectors ({build_time:.1f}s)", flush=True)
    if cache_path:
        torch.save({"embeddings": coarse_embeds, "faiss_to_coarse": faiss_to_coarse}, cache_path)
        print(f"Saved coarse embedding cache: {cache_path}", flush=True)
    return index, faiss_to_coarse, build_time


def build_faiss_ranking(query, encoder, device, faiss_index, faiss_to_coarse):
    start = time.perf_counter()
    zq = get_graph_embedding(query.to(device), encoder, device)
    _, indices = faiss_index.search(zq.detach().cpu().numpy(), faiss_index.ntotal)
    ranking = [
        int(faiss_to_coarse.get(int(index), int(index)))
        for index in indices[0]
        if int(index) >= 0
    ]
    return ranking, time.perf_counter() - start


def build_coarse_topo_features(data, hierarchy, adj_t, cache_dir, cache_key):
    path = os.path.join(cache_dir, f"{cache_key}_coarse_topo_features.pt") if cache_dir else ""
    if path and os.path.exists(path):
        cached = torch.load(path, map_location="cpu")
        return cached["features"], cached["ids"], cached["mean"], cached["std"]
    ids = sorted(int(pid) for pid in hierarchy["coarse_part_node_sets"].keys())
    rows = []
    overlap_sets = hierarchy.get("coarse_overlap_node_sets", {})
    for pid in ids:
        nodes = sorted(int(n) for n in hierarchy["coarse_part_node_sets"].get(pid, ()))
        n = len(nodes)
        if n:
            node_tensor = torch.tensor(nodes, dtype=torch.long)
            e = int(adj_t[node_tensor, node_tensor].nnz())
        else:
            e = 0
        overlap_n = len(overlap_sets.get(pid, ()))
        avg_degree = float(e) / max(float(n), 1.0)
        density = float(e) / max(float(n * max(n - 1, 1)), 1.0)
        boundary_ratio = float(max(overlap_n - n, 0)) / max(float(n), 1.0)
        rows.append([math.log1p(n), math.log1p(e), avg_degree, density, boundary_ratio])
    feats = torch.tensor(rows, dtype=torch.float32)
    mu = feats.mean(dim=0, keepdim=True)
    sigma = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
    feats = (feats - mu) / sigma
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    if path:
        torch.save({"features": feats, "ids": ids, "mean": mu, "std": sigma}, path)
    return feats, ids, mu, sigma


def rank_by_topo_feature(query, coarse_topo_features, coarse_topo_ids, topo_mean, topo_std):
    n = int(query.num_nodes)
    e = int(query.edge_index.size(1)) if query.edge_index is not None else 0
    avg_degree = float(e) / max(float(n), 1.0)
    density = float(e) / max(float(n * max(n - 1, 1)), 1.0)
    q = torch.tensor([[math.log1p(n), math.log1p(e), avg_degree, density, 0.0]], dtype=torch.float32)
    q = (q - topo_mean) / topo_std
    q = torch.nn.functional.normalize(q, p=2, dim=1)
    scores = torch.mm(q, coarse_topo_features.t()).squeeze(0)
    order = torch.argsort(scores, descending=True).tolist()
    return [int(coarse_topo_ids[i]) for i in order]


def build_coarse_feature_index(data, hierarchy, cache_dir, cache_key, node_labels_override=None):
    """Classical filter-and-verify feature index (GraphGrep/gIndex style).

    Each coarse partition is summarized by the set of node-label tokens it
    contains and the set of internal edge label-pairs (label(u),label(v)) with
    u,v in the same partition. A partition can contain the query only if its
    feature sets superset the query's (a necessary containment condition); we
    rank partitions by how many of the query's features they cover. This is the
    canonical structural-feature index baseline, distinct from the learned and
    mean-feature retrievers.
    """
    tag = "class" if node_labels_override is not None else "feature"
    path = os.path.join(cache_dir, f"{cache_key}_coarse_feature_index_{tag}.pt") if cache_dir else ""
    if path and os.path.exists(path):
        cached = torch.load(path, map_location="cpu", weights_only=False)
        return cached["node_labels"], cached["edge_pairs"], cached["ids"]
    print(f"Building coarse feature index ({tag} labels)...", flush=True)
    ids = sorted(int(pid) for pid in hierarchy["coarse_part_node_sets"].keys())
    node_labels = {int(i): torch.tensor([], dtype=torch.long) for i in ids}
    # per-node label
    if node_labels_override is not None:
        labels = node_labels_override.detach().cpu().long().view(-1)
    else:
        labels = torch.tensor(
            [feature_to_label(data.x[i]) for i in range(data.num_nodes)], dtype=torch.long
        )
    # node-label sets per partition
    for pid in ids:
        nodes = list(hierarchy["coarse_part_node_sets"].get(pid, ()))
        if nodes:
            node_labels[pid] = torch.unique(labels[torch.tensor(nodes, dtype=torch.long)])
    # internal edge label-pairs per partition (single pass over edges)
    part_ids = torch.full((data.num_nodes,), -1, dtype=torch.long)
    for node_id, p in hierarchy["node_to_coarse_map"].items():
        part_ids[int(node_id)] = int(p)
    src, dst = data.edge_index.detach().cpu().long()
    ps, pd = part_ids[src], part_ids[dst]
    internal = (ps == pd) & (ps >= 0)
    idx = torch.nonzero(internal, as_tuple=False).flatten()
    la, lb = labels[src[idx]], labels[dst[idx]]
    lo = torch.minimum(la, lb)
    hi = torch.maximum(la, lb)
    pair_key = lo * (int(labels.max()) + 1) + hi  # encode unordered label pair
    edge_pairs = {}
    home = ps[idx]
    for pid in ids:
        m = home == pid
        edge_pairs[int(pid)] = torch.unique(pair_key[m]) if int(m.sum()) else torch.tensor([], dtype=torch.long)
    if path:
        torch.save({"node_labels": node_labels, "edge_pairs": edge_pairs, "ids": ids,
                    "label_base": int(labels.max()) + 1}, path)
    return node_labels, edge_pairs, ids


def rank_by_feature_index(query, feature_index, query_labels=None):
    node_labels, edge_pairs, ids = feature_index
    if query_labels is not None:
        qlabels = torch.unique(torch.tensor([int(x) for x in query_labels], dtype=torch.long))
    else:
        qlabels = torch.unique(torch.tensor(
            [feature_to_label(query.x[i]) for i in range(query.num_nodes)], dtype=torch.long
        ))
    scored = []
    for pid in ids:
        nl = node_labels.get(int(pid))
        node_cov = int(torch.isin(qlabels, nl).sum()) if nl is not None and nl.numel() else 0
        scored.append((node_cov, int(pid)))
    # rank by node-label coverage desc (the necessary containment signal),
    # tie-broken by partition id for determinism
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [pid for _cov, pid in scored]


def load_or_build_coarse_mean_features(data, hierarchy, cache_dir, cache_key):
    path = os.path.join(cache_dir, f"{cache_key}_coarse_mean_features.pt") if cache_dir else ""
    if path and os.path.exists(path):
        print(f"Loading cached coarse mean features: {path}", flush=True)
        return torch_load_any(path)
    print("Building coarse mean features with vectorized index_add...", flush=True)
    start = time.perf_counter()
    num_parts = len(hierarchy["coarse_part_nodes_map"])
    features = data.x.detach().cpu().float()
    pairs = [
        (int(node_id), int(part_id))
        for node_id, part_id in hierarchy["node_to_coarse_map"].items()
        if 0 <= int(part_id) < num_parts
    ]
    node_ids = torch.tensor([node for node, _ in pairs], dtype=torch.long)
    part_ids = torch.tensor([part for _, part in pairs], dtype=torch.long)
    sums = torch.zeros((num_parts, features.size(1)), dtype=torch.float32)
    counts = torch.zeros((num_parts,), dtype=torch.float32)
    sums.index_add_(0, part_ids, features[node_ids])
    counts.index_add_(0, part_ids, torch.ones_like(part_ids, dtype=torch.float32))
    means = torch.nn.functional.normalize(
        sums / counts.clamp_min(1.0).unsqueeze(1), p=2, dim=1
    )
    if path:
        torch.save(means, path)
        print(
            f"Saved coarse mean features: {path} ({time.perf_counter() - start:.1f}s)",
            flush=True,
        )
    return means


def candidate_nodes_for_parts(selected_ids, hierarchy, use_overlap=True):
    part_sets = hierarchy["coarse_part_node_sets"]
    overlap_sets = hierarchy.get("coarse_overlap_node_sets", {})
    nodes = set()
    for part_id in selected_ids:
        part_id = int(part_id)
        nodes.update(part_sets.get(part_id, ()))
        if use_overlap:
            nodes.update(overlap_sets.get(part_id, ()))
    return nodes


# --- selective overlap -------------------------------------------------------
#
# The blunt one-hop overlap operator (above) unions *every* boundary-touching
# node from *every* neighbor partition. On MAG a single coarse part borders ~948
# of 2,000 partitions, so the union explodes to ~44-83% of the graph before
# pruning. Selective overlap instead expands each selected partition to only its
# strongest / most relevant neighbors, derived entirely from the cached overlap
# index (no edge_index needed): a node v in part p's overlap has a "home"
# partition (the part whose node set contains it), so p's overlap groups by
# contributing neighbor partition, weighted by boundary-node support.

def build_overlap_neighbor_index(hierarchy):
    """Group each coarse part's one-hop overlap by contributing neighbor partition.

    Adds two structures to ``hierarchy`` (idempotent):
      coarse_overlap_by_neighbor[p] = {neighbor_part_q: frozenset(overlap_nodes)}
      coarse_overlap_support[p]      = {neighbor_part_q: support_count}
    where ``support_count`` is the number of boundary nodes neighbor q contributes
    to part p -- the bridge-evidence weight used to rank neighbors.
    """
    if (
        "coarse_overlap_by_neighbor" in hierarchy
        and "coarse_overlap_support" in hierarchy
        and "coarse_overlap_neighbor_ranked" in hierarchy
    ):
        return hierarchy
    part_sets = hierarchy["coarse_part_node_sets"]
    overlap_sets = hierarchy.get("coarse_overlap_node_sets", {})
    print("Building selective-overlap neighbor index...", flush=True)
    start = time.perf_counter()
    home = {}
    for part_id, nodes in part_sets.items():
        part_id = int(part_id)
        for node in nodes:
            home[int(node)] = part_id
    by_neighbor = {}
    support = {}
    ranked = {}
    for part_id, onodes in overlap_sets.items():
        part_id = int(part_id)
        groups = defaultdict(list)
        for node in onodes:
            node = int(node)
            neighbor = home.get(node)
            if neighbor is not None and neighbor != part_id:
                groups[neighbor].append(node)
        frozen = {q: frozenset(vs) for q, vs in groups.items()}
        by_neighbor[part_id] = frozen
        support[part_id] = {q: len(vs) for q, vs in frozen.items()}
        # Precompute the support-descending neighbor order once so per-query
        # selective expansion never has to re-sort.
        ranked[part_id] = sorted(
            ((len(vs), q, vs) for q, vs in frozen.items()),
            key=lambda item: item[0],
            reverse=True,
        )
    hierarchy["coarse_overlap_by_neighbor"] = by_neighbor
    hierarchy["coarse_overlap_support"] = support
    hierarchy["coarse_overlap_neighbor_ranked"] = ranked
    print(
        f"Built selective-overlap neighbor index for {len(by_neighbor):,} parts "
        f"in {time.perf_counter() - start:.1f}s.",
        flush=True,
    )
    return hierarchy


def load_or_build_overlap_neighbor_index(hierarchy, cache_dir, cache_key):
    if not cache_dir:
        return build_overlap_neighbor_index(hierarchy)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{cache_key}_overlap_neighbor_index.pt")
    if "coarse_overlap_neighbor_ranked" not in hierarchy and os.path.exists(path):
        print(f"Loading cached overlap neighbor index: {path}", flush=True)
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if "coarse_overlap_neighbor_ranked" in cached:
            hierarchy.update(cached)
            return hierarchy
        # Legacy cache without the ranked order; rebuild to add it.
        hierarchy.update(cached)
    hierarchy = build_overlap_neighbor_index(hierarchy)
    if not os.path.exists(path):
        torch.save(
            {
                "coarse_overlap_by_neighbor": hierarchy["coarse_overlap_by_neighbor"],
                "coarse_overlap_support": hierarchy["coarse_overlap_support"],
                "coarse_overlap_neighbor_ranked": hierarchy["coarse_overlap_neighbor_ranked"],
            },
            path,
        )
        print(f"Saved overlap neighbor index cache: {path}", flush=True)
    return hierarchy


def overlap_policy_is_active(policy):
    """True if the policy modifies overlap beyond the blunt union."""
    if not policy:
        return False
    return any(
        policy.get(key) is not None
        for key in ("max_parts", "min_support", "max_nodes", "bridge_infill_top")
    ) or bool(policy.get("label_compatible"))


def _label_filter_nodes(node_iterable, query, label_tokens, query_labels=None):
    """Keep only nodes whose label token matches a query label.

    query_labels (the query's per-node labels under the active label source) is
    preferred; falling back to the feature hash only when not supplied.
    """
    if query_labels is None:
        query_labels = [feature_to_label(query.x[i]) for i in range(query.num_nodes)]
    query_labels = torch.tensor(sorted(set(int(x) for x in query_labels)), dtype=torch.long)
    node_tensor = torch.tensor(sorted(int(n) for n in node_iterable), dtype=torch.long)
    keep_mask = torch.isin(label_tokens[node_tensor], query_labels)
    return set(int(n) for n in node_tensor[keep_mask].tolist())


def bridge_infill_nodes(selected_ids, hierarchy, top_m, min_support=0):
    """Recall-expansion for dispersed selections (e.g. random walks).

    Adds the full partition nodes of the ``top_m`` partitions most strongly bridged
    to the selected set, ranked by total boundary support (the number of boundary
    nodes they contribute across selected partitions). Motivation: a random walk
    visits A->M->B with consecutive partitions graph-adjacent, so a missed mid-path
    partition M is strongly bridged to the selected walk partitions; one-hop
    boundary overlap reaches only M's boundary nodes, while this recovers M's
    interior. We rank by *support* rather than binary adjacency because MAG's coarse
    quotient graph is near-complete -- almost every partition is adjacent to a large
    selected set, so adjacency alone is not selective; support is. Bounded by
    ``top_m`` so it cannot explode regardless of selection size. Read from the cached
    neighbor index (no edge_index needed).
    """
    support = hierarchy.get("coarse_overlap_support")
    part_sets = hierarchy["coarse_part_node_sets"]
    selected_set = set(int(p) for p in selected_ids)
    total_support = defaultdict(int)
    for part_id in selected_ids:
        for neighbor, sup in support.get(int(part_id), {}).items():
            if neighbor not in selected_set:
                total_support[neighbor] += sup
    ranked = sorted(total_support.items(), key=lambda kv: kv[1], reverse=True)
    out = set()
    added = 0
    for neighbor, sup in ranked:
        if added >= top_m or sup < min_support:
            break
        out.update(part_sets.get(neighbor, ()))
        added += 1
    return out


def selective_overlap_for_parts(
    selected_ids,
    hierarchy,
    policy=None,
    query=None,
    label_tokens=None,
    query_labels=None,
):
    """Candidate nodes = selected partitions + a *selective* one-hop overlap.

    policy keys (all optional; absent/None => no restriction on that axis):
      use_overlap   : bool   include overlap at all (default True)
      max_parts     : int    per selected part, keep only its top-N neighbor parts
                             by boundary support
      min_support   : int    drop neighbor parts contributing fewer than N nodes
      max_nodes     : int    global cap on total overlap nodes added (greedy by
                             aggregated neighbor support)
      label_compatible: bool only add overlap nodes whose label token matches a
                             query label (front-loads coverage-lossless pruning)
      boundary_overlap: bool include the one-hop boundary overlap at all (default
                             True; set False for bridge-infill-only experiments)
      bridge_infill_top: int add full partition nodes of the top-N partitions most
                             strongly bridged (by boundary support) to the selected
                             set (recall-expansion for dispersed/random-walk queries)
      bridge_infill_min_support: int minimum total support for a bridge candidate

    With an empty/None policy this reduces to the blunt union (matches
    candidate_nodes_for_parts), so it is a safe drop-in.
    """
    policy = policy or {}
    part_sets = hierarchy["coarse_part_node_sets"]
    label_compatible = bool(policy.get("label_compatible"))
    can_label = label_compatible and query is not None and label_tokens is not None and query.x is not None
    nodes = set()
    for part_id in selected_ids:
        nodes.update(part_sets.get(int(part_id), ()))

    if not policy.get("use_overlap", True):
        return nodes

    # Fast path: blunt union, no neighbor index required.
    if not overlap_policy_is_active(policy):
        overlap_sets = hierarchy.get("coarse_overlap_node_sets", {})
        for part_id in selected_ids:
            nodes.update(overlap_sets.get(int(part_id), ()))
        return nodes

    if hierarchy.get("coarse_overlap_neighbor_ranked") is None:
        raise RuntimeError(
            "selective overlap requires the neighbor index; call "
            "build_overlap_neighbor_index(hierarchy) first."
        )

    # --- one-hop boundary overlap (optionally selective by support/labels) ---
    if policy.get("boundary_overlap", True):
        ranked_index = hierarchy["coarse_overlap_neighbor_ranked"]
        max_parts = policy.get("max_parts")
        min_support = policy.get("min_support")
        max_nodes = policy.get("max_nodes")
        # Aggregate each candidate neighbor's best support across selected seeds;
        # neighbor order is precomputed (support-descending) so there is no sort.
        node_best_support = {}
        for part_id in selected_ids:
            kept = 0
            for sup, _neighbor, vs in ranked_index.get(int(part_id), ()):
                if min_support is not None and sup < min_support:
                    break  # ranked desc: everything after is smaller too
                if max_parts is not None and kept >= max_parts:
                    break
                for node in vs:
                    node = int(node)
                    if sup > node_best_support.get(node, 0):
                        node_best_support[node] = sup
                kept += 1
        if max_nodes is not None and len(node_best_support) > max_nodes:
            ordered = sorted(node_best_support.items(), key=lambda kv: kv[1], reverse=True)
            node_best_support = dict(ordered[:max_nodes])
        if can_label and node_best_support:
            nodes.update(_label_filter_nodes(node_best_support, query, label_tokens, query_labels=query_labels))
        else:
            nodes.update(node_best_support)

    # --- bridge infill (recall-expansion for dispersed selections) ---
    bridge_top = policy.get("bridge_infill_top")
    if bridge_top:
        infill = bridge_infill_nodes(
            selected_ids, hierarchy, int(bridge_top), int(policy.get("bridge_infill_min_support", 0))
        )
        if can_label and infill:
            infill = _label_filter_nodes(infill, query, label_tokens, query_labels=query_labels)
        nodes.update(infill)

    return nodes


def prune_nodes_by_signature(nodes, query, signature_tokens, signature_name):
    if signature_tokens is None or not nodes:
        return nodes
    # Keep accumulated nodes whose attribute signature matches a query node.
    # Query tokens come from the query payload, not planted target/global IDs.
    query_tokens = derive_query_signature_tokens(query, signature_name)
    node_tensor = torch.tensor(sorted(int(node) for node in nodes), dtype=torch.long)
    keep = torch.isin(signature_tokens[node_tensor].long(), query_tokens)
    return set(node_tensor[keep].tolist())


def prune_nodes_by_query_label_tokens(nodes, query, label_tokens, query_labels=None):
    if not nodes or label_tokens is None:
        return nodes
    if query_labels is None:
        if query.x is None:
            return nodes
        query_labels = [feature_to_label(query.x[i]) for i in range(query.num_nodes)]
    n_query = len(query_labels)
    query_labels = torch.tensor(sorted(set(int(x) for x in query_labels)), dtype=torch.long)
    node_tensor = torch.tensor(sorted(int(node) for node in nodes), dtype=torch.long)
    keep_mask = torch.isin(label_tokens[node_tensor], query_labels)
    kept = set(int(node) for node in node_tensor[keep_mask].tolist())
    return kept if len(kept) >= n_query else nodes



def node_fullcov(query_nodes, candidate_nodes):
    query_set = {int(node) for node in query_nodes}
    missed = sorted(query_set - candidate_nodes)
    return bool(query_set) and not missed, len(missed)


def component_diagnostics(nodes, adj_t):
    comps = connected_components(nodes, adj_t)
    if not comps:
        return 0, 0
    return len(comps), max(len(comp) for comp in comps)


def connected_components(nodes, adj_t):
    if not nodes:
        return []
    node_tensor = torch.tensor(sorted(int(n) for n in nodes), dtype=torch.long)
    sub_adj = adj_t[node_tensor, node_tensor]
    row, col, _ = sub_adj.coo()
    parent = list(range(int(node_tensor.numel())))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in zip(row.tolist(), col.tolist()):
        union(int(u), int(v))
    groups = {}
    for i in range(int(node_tensor.numel())):
        root = find(i)
        groups.setdefault(root, []).append(int(node_tensor[i].item()))
    return list(groups.values())


def query_label_counts(query, query_labels=None):
    if query_labels is None:
        query_labels = [feature_to_label(query.x[i]) for i in range(query.num_nodes)]
    counts = {}
    for label in query_labels:
        label = int(label)
        counts[label] = counts.get(label, 0) + 1
    return counts


def component_covers_query_labels(component_nodes, label_tokens, query_counts):
    if len(component_nodes) < sum(query_counts.values()):
        return False
    node_tensor = torch.tensor(component_nodes, dtype=torch.long)
    values = label_tokens[node_tensor].tolist()
    counts = {}
    for value in values:
        value = int(value)
        if value in query_counts:
            counts[value] = counts.get(value, 0) + 1
    return all(counts.get(label, 0) >= need for label, need in query_counts.items())


def _adjacency_lists(edge_index, num_nodes):
    adj = [[] for _ in range(int(num_nodes))]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(src, dst):
        u, v = int(u), int(v)
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    return [sorted(set(items)) for items in adj]


def _random_walk_query(data, adj_t, adj_lists, target_size):
    min_nodes = max(5, target_size - 10)
    for _ in range(80):
        cur = random.randrange(int(data.num_nodes))
        order = [cur]
        seen = {cur}
        for _step in range(target_size * 20):
            neigh = adj_lists[cur]
            if not neigh:
                cur = random.choice(order)
                continue
            cur = random.choice(neigh)
            if cur not in seen:
                seen.add(cur)
                order.append(cur)
                if len(order) >= target_size:
                    break
        if len(order) < min_nodes:
            continue
        nodes = bench._tensor_from_ordered_nodes(order[:target_size])
        query = bench.extract_subgraph(adj_t, nodes, data)
        if query is not None and query.num_nodes >= 5 and query.num_edges >= 2:
            return query, nodes, None, "random_walk"
    return None


def _degree_k_hop_query(data, adj_t, degrees, target_size, attempt):
    min_nodes = max(5, target_size - 10)
    nonzero = torch.nonzero(degrees > 0, as_tuple=False).view(-1)
    if int(nonzero.numel()) == 0:
        return None
    sorted_nodes = nonzero[torch.argsort(degrees[nonzero])]
    third = max(1, int(sorted_nodes.numel()) // 3)
    bins = [sorted_nodes[:third], sorted_nodes[third:2 * third], sorted_nodes[2 * third:]]
    pool = bins[attempt % 3]
    if int(pool.numel()) == 0:
        pool = sorted_nodes
    for _ in range(50):
        anchor = int(pool[random.randrange(int(pool.numel()))].item())
        try:
            subset, _, _, _ = bench.k_hop_subgraph(
                anchor,
                num_hops=3,
                edge_index=data.edge_index,
                relabel_nodes=False,
                num_nodes=data.num_nodes,
            )
        except Exception:
            continue
        if len(subset) < min_nodes:
            continue
        nodes = bench._connected_bfs_nodes(
            adj_t,
            anchor,
            target_size,
            allowed_nodes=subset,
            min_nodes=min_nodes,
        )
        if nodes is None:
            continue
        query = bench.extract_subgraph(adj_t, nodes, data)
        if query is not None and query.num_nodes >= 5 and query.num_edges >= 2:
            return query, nodes, None, "degree_k_hop"
    return None


def _label_corrupt_negative(query, seed_value):
    neg = query.clone()
    if neg.x is None or neg.num_nodes <= 0:
        return None
    x = neg.x.detach().clone()
    row = int(seed_value) % int(neg.num_nodes)
    x[row] = torch.zeros_like(x[row])
    if x.dim() == 2 and x.size(1) > 0:
        x[row, 0] = 1234567.0 + float(seed_value % 100000)
    neg.x = x
    return neg


def _structure_corrupt_negative(query, seed_value):
    neg = query.clone()
    if neg.num_nodes < 2:
        return None
    edges = set()
    if neg.edge_index is not None and neg.edge_index.numel() > 0:
        for u, v in zip(neg.edge_index[0].tolist(), neg.edge_index[1].tolist()):
            u, v = int(u), int(v)
            if u != v:
                edges.add((min(u, v), max(u, v)))
    pairs = [(i, j) for i in range(neg.num_nodes) for j in range(i + 1, neg.num_nodes)]
    rng = random.Random(int(seed_value))
    rng.shuffle(pairs)
    for u, v in pairs:
        key = (min(u, v), max(u, v))
        if key in edges:
            continue
        add = torch.tensor([[u, v], [v, u]], dtype=torch.long)
        neg.edge_index = torch.cat([neg.edge_index.detach().cpu(), add], dim=1)
        if hasattr(neg, "edge_type") and neg.edge_type is not None:
            add_type = torch.zeros(add.size(1), dtype=torch.long)
            neg.edge_type = torch.cat([neg.edge_type.detach().cpu().long(), add_type], dim=0)
        return neg
    return None


def generate_cascade_queries(data, hierarchy, count, target_size, seed, query_types):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    edge_values = data.edge_type.long() if hasattr(data, "edge_type") and data.edge_type is not None else None
    adj_t = SparseTensor.from_edge_index(
        data.edge_index, edge_attr=edge_values, sparse_sizes=(data.num_nodes, data.num_nodes)
    )
    adj_lists = _adjacency_lists(data.edge_index, data.num_nodes)
    degrees = torch.zeros(int(data.num_nodes), dtype=torch.long)
    for node, neigh in enumerate(adj_lists):
        degrees[node] = len(neigh)
    requested = [part.strip() for part in query_types.split(",") if part.strip()]
    if requested == ["all"]:
        requested = [
            "single", "multi_fine", "multi_coarse", "k_hop",
            "random_walk", "degree_k_hop", "negative_label", "negative_structure",
        ]
    if requested == ["positive"]:
        requested = ["single", "multi_fine", "multi_coarse", "k_hop", "random_walk", "degree_k_hop"]
    if requested == ["negative"]:
        requested = ["negative_label", "negative_structure"]
    coarse_ids = sorted(int(x) for x in hierarchy["coarse_part_nodes_map"].keys())
    queries = []
    for qtype in requested:
        made = 0
        attempts = 0
        while made < count and attempts < count * 50:
            attempts += 1
            if qtype == "k_hop":
                generated = bench.generate_k_hop_query(data, adj_t, target_size=target_size)
            elif qtype == "single":
                generated = bench.generate_single_partition_query(
                    data, adj_t, hierarchy["coarse_part_nodes_map"],
                    random.choice(coarse_ids), target_size=target_size
                )
            elif qtype == "multi_fine":
                generated = bench.generate_multi_fine_query(
                    data, adj_t, None, hierarchy["fine_part_nodes_map"],
                    hierarchy["fine_to_coarse_map"], random.choice(coarse_ids),
                    target_size=target_size
                )
            elif qtype == "multi_coarse":
                generated = bench.generate_multi_coarse_query(
                    data, adj_t, hierarchy["coarse_part_nodes_map"],
                    hierarchy["coarse_part_graph"], hierarchy["fine_part_nodes_map"],
                    hierarchy["fine_to_coarse_map"], target_size=target_size
                )
            elif qtype == "random_walk":
                generated = _random_walk_query(data, adj_t, adj_lists, target_size)
            elif qtype in {"degree_k_hop", "degree_stratified_k_hop"}:
                generated = _degree_k_hop_query(data, adj_t, degrees, target_size, attempts)
            elif qtype in {"negative_label", "label_corrupt_negative"}:
                generated = bench.generate_k_hop_query(data, adj_t, target_size=target_size)
                if generated is not None:
                    query, query_nodes, true_coarse, _ = generated
                    query = _label_corrupt_negative(query, seed + made + attempts)
                    generated = (query, query_nodes, true_coarse, "negative_label") if query is not None else None
            elif qtype in {"negative_structure", "structure_corrupt_negative"}:
                generated = bench.generate_k_hop_query(data, adj_t, target_size=target_size)
                if generated is not None:
                    query, query_nodes, true_coarse, _ = generated
                    query = _structure_corrupt_negative(query, seed + made + attempts)
                    generated = (query, query_nodes, true_coarse, "negative_structure") if query is not None else None
            else:
                raise ValueError(f"Unknown query type: {qtype}")
            if generated is None:
                continue
            query, query_nodes, true_coarse, _ = generated
            if not bench._subgraph_is_connected(query):
                continue
            if not true_coarse:
                true_coarse = bench.determine_true_coarse(query_nodes, hierarchy["node_to_coarse_map"])
            true_fine = bench.determine_true_fine(query_nodes, hierarchy["node_to_fine_map"])
            if not true_coarse or not true_fine:
                continue
            queries.append({
                "query_id": f"{qtype}_q{target_size}_{made}",
                "query_type": qtype,
                "is_negative": qtype in {"negative_label", "label_corrupt_negative", "negative_structure", "structure_corrupt_negative"},
                "negative_type": qtype if qtype.startswith("negative") or qtype.endswith("_negative") else "",
                "expected_match": qtype not in {"negative_label", "label_corrupt_negative", "negative_structure", "structure_corrupt_negative"},
                "target_query_size": int(target_size),
                "query": query,
                "query_nodes": query_nodes.detach().cpu().long(),
                "true_coarse": set(int(x) for x in true_coarse),
                "true_fine": set(int(x) for x in true_fine),
            })
            made += 1
        if made != count:
            raise RuntimeError(f"Generated only {made}/{count} {qtype} queries")
    print(f"Generated {len(queries)} fixed-seed queries: {','.join(requested)}.", flush=True)
    return queries


def torch_load_any(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _expanded_query_types(query_types):
    requested = [part.strip() for part in str(query_types).split(",") if part.strip()]
    if requested == ["all"]:
        return [
            "single", "multi_fine", "multi_coarse", "k_hop",
            "random_walk", "degree_k_hop", "negative_label", "negative_structure",
        ]
    if requested == ["positive"]:
        return ["single", "multi_fine", "multi_coarse", "k_hop", "random_walk", "degree_k_hop"]
    if requested == ["negative"]:
        return ["negative_label", "negative_structure"]
    return requested


def _queries_match_spec(queries, count, sizes, query_types):
    expected_types = set(_expanded_query_types(query_types))
    expected_sizes = set(int(s) for s in sizes)
    expected_total = int(count) * len(expected_types) * len(expected_sizes)
    if len(queries) != expected_total:
        return False
    seen_types = {str(q.get("query_type", "")) for q in queries}
    seen_sizes = {int(q.get("target_query_size", -1)) for q in queries}
    if seen_types != expected_types or seen_sizes != expected_sizes:
        return False
    counts = {}
    for q in queries:
        key = (int(q.get("target_query_size", -1)), str(q.get("query_type", "")))
        counts[key] = counts.get(key, 0) + 1
        query = q.get("query")
        if query is None or not bench._subgraph_is_connected(query):
            return False
    return all(v == int(count) for v in counts.values())


def _load_manifest_queries(cache_dir, seed, count, sizes, query_types):
    if not cache_dir:
        return None
    manifest_path = os.path.join(cache_dir, "query_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        print(f"[CACHE] Failed to read query manifest {manifest_path}: {exc}", flush=True)
        return None
    spec_key = f"{seed}|{count}|{','.join(str(s) for s in sizes)}|{query_types}"
    filename = manifest.get(spec_key) or manifest.get(str(seed))
    if not filename:
        return None
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        print(f"[CACHE] Manifest query cache missing: {path}", flush=True)
        return None
    cached = torch_load_any(path)
    if not _queries_match_spec(cached, count, sizes, query_types):
        print(f"[CACHE] Manifest query cache spec mismatch: {path}", flush=True)
        return None
    print(f"Loading manifest fixed queries: {path}", flush=True)
    print(f"Loaded {len(cached)} manifest fixed queries.", flush=True)
    return cached


def _update_query_manifest(cache_dir, seed, count, sizes, query_types, cache_path):
    if not cache_dir or not cache_path:
        return
    manifest_path = os.path.join(cache_dir, "query_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            manifest = {}
    filename = os.path.basename(cache_path)
    manifest[str(seed)] = filename
    manifest[f"{seed}|{count}|{','.join(str(s) for s in sizes)}|{query_types}"] = filename
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, manifest_path)
    print(f"Saved query manifest: {manifest_path}", flush=True)


def load_or_generate_cascade_queries(
    data,
    hierarchy,
    count,
    target_sizes,
    seed,
    query_types,
    cache_dir,
    cache_key,
):
    sizes = parse_budgets(target_sizes)
    query_key = safe_cache_key(
        cache_key,
        "queries_v3_connected",
        count,
        ",".join(str(s) for s in sizes),
        seed,
        query_types,
    )
    cache_path = os.path.join(cache_dir, f"{query_key}_queries.pt") if cache_dir else ""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached fixed queries: {cache_path}", flush=True)
        cached = torch_load_any(cache_path)
        print(f"Loaded {len(cached)} cached fixed queries.", flush=True)
        if _queries_match_spec(cached, count, sizes, query_types):
            return cached
        print("Cached fixed queries do not match the current connected-query spec; rebuilding.", flush=True)

    manifest_cached = _load_manifest_queries(cache_dir, seed, count, sizes, query_types)
    if manifest_cached is not None:
        return manifest_cached

    queries = []
    for target_size in sizes:
        queries.extend(
            generate_cascade_queries(
                data, hierarchy, count, target_size, seed, query_types
            )
        )
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(queries, cache_path)
        print(f"Saved fixed query cache: {cache_path}", flush=True)
        _update_query_manifest(cache_dir, seed, count, sizes, query_types, cache_path)
    return queries


def select_ids_for_budget(method, ranking, budget, hierarchy, seed_count, query_id=""):
    if method == "all":
        return sorted(int(part_id) for part_id in hierarchy["coarse_part_nodes_map"])
    if method in {"fixed", "mean_feature", "coarse_mean_rrf", "topo_feature", "feature_index"}:
        return ranking[:budget]
    if method == "random":
        ids = sorted(int(part_id) for part_id in hierarchy["coarse_part_nodes_map"])
        seed = int(hashlib.md5(str(query_id).encode("utf-8")).hexdigest()[:8], 16)
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(ids), generator=generator).tolist()
        return [ids[i] for i in perm[: min(budget, len(ids))]]
    if method == "hybrid":
        if budget <= seed_count:
            return ranking[:budget]
        return hybrid_boundary_expand(
            ranking[:seed_count],
            ranking,
            budget,
            hierarchy["coarse_part_graph"],
            seed_count=seed_count,
            model_weight=0.5,
            teleport_every=10,
        )
    raise ValueError(f"Unknown method: {method}")


def method_needs_encoder(method):
    return method in {"fixed", "hybrid", "coarse_mean_rrf"}


def dump_partition_store(
    out_dir,
    data,
    hierarchy,
    tokens,
    label_tokens,
    node_label_full,
    adj_t,
    faiss_index,
    faiss_to_coarse,
    coarse_embeddings_or_None,
    meta,
):
    """Serialize a per-partition on-disk store for streaming/out-of-core serving.

    For each coarse partition p in hierarchy["coarse_part_node_sets"], writes
    ``out_dir/parts/{p}.pt`` = {
        "nodes": LongTensor of sorted expanded GLOBAL ids
                 (coarse_part_node_sets[p] UNION coarse_overlap_node_sets[p]),
        "edge_index": LongTensor[2, E] of induced edges among those nodes,
                      endpoints stored as GLOBAL ids,
    }
    plus store-wide artifacts: signature_tokens.pt, label_tokens.pt,
    coarse_embeddings.pt (to rebuild an IndexFlatL2 without the full graph),
    and meta.json.

    Robust to tokens / labels / embeddings being None.
    """
    parts_dir = os.path.join(out_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    part_sets = hierarchy["coarse_part_node_sets"]
    overlap_sets = hierarchy.get("coarse_overlap_node_sets", {})

    part_ids = sorted(int(p) for p in part_sets.keys())
    written = 0
    for pid in part_ids:
        node_set = set(int(n) for n in part_sets.get(pid, ()))
        node_set.update(int(n) for n in overlap_sets.get(pid, ()))
        if not node_set:
            torch.save(
                {
                    "nodes": torch.zeros(0, dtype=torch.long),
                    "edge_index": torch.zeros(2, 0, dtype=torch.long),
                },
                os.path.join(parts_dir, f"{pid}.pt"),
            )
            continue
        node_tensor = torch.tensor(sorted(node_set), dtype=torch.long)
        # Induced edges among the node set. extract_subgraph returns edge_index in
        # LOCAL (0..k-1) indices with sub.global_id giving the mapping back to
        # global ids; remap local endpoints to global ids for storage.
        sub = bench.extract_subgraph(adj_t, node_tensor, data)
        if sub is None or sub.edge_index is None or sub.edge_index.numel() == 0:
            global_edge_index = torch.zeros(2, 0, dtype=torch.long)
        else:
            gid = sub.global_id.detach().cpu().long()
            local_edge = sub.edge_index.detach().cpu().long()
            global_edge_index = gid[local_edge]
        torch.save(
            {"nodes": node_tensor, "edge_index": global_edge_index},
            os.path.join(parts_dir, f"{pid}.pt"),
        )
        written += 1

    # signature tokens (per-node LongTensor or a None marker)
    torch.save(
        {"tokens": tokens.detach().cpu() if tokens is not None else None},
        os.path.join(out_dir, "signature_tokens.pt"),
    )

    # per-node label tokens: prefer node_label_full, else label_tokens
    labels_out = node_label_full if node_label_full is not None else label_tokens
    torch.save(
        {"labels": labels_out.detach().cpu().long() if labels_out is not None else None},
        os.path.join(out_dir, "label_tokens.pt"),
    )

    # coarse embeddings sufficient to rebuild an IndexFlatL2 without the full graph
    embeddings = None
    if coarse_embeddings_or_None is not None:
        embeddings = coarse_embeddings_or_None
    elif faiss_index is not None:
        try:
            # faiss IndexFlatL2 exposes stored vectors via reconstruct_n; the
            # NumpyIndexFlatL2 fallback stores them on `.vectors`.
            if hasattr(faiss_index, "reconstruct_n"):
                embeddings = torch.from_numpy(
                    faiss_index.reconstruct_n(0, faiss_index.ntotal)
                ).float()
            elif hasattr(faiss_index, "vectors"):
                embeddings = torch.from_numpy(
                    np.asarray(faiss_index.vectors, dtype=np.float32)
                ).float()
        except Exception as exc:
            print(f"Could not reconstruct coarse embeddings from faiss index: {exc}", flush=True)
            embeddings = None
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().float()
    torch.save(
        {
            "embeddings": embeddings,
            "faiss_to_coarse": {int(k): int(v) for k, v in (faiss_to_coarse or {}).items()},
        },
        os.path.join(out_dir, "coarse_embeddings.pt"),
    )

    meta_out = dict(meta or {})
    meta_out.setdefault("num_nodes", int(data.num_nodes))
    meta_out.setdefault("num_coarse", len(part_ids))
    if isinstance(embeddings, torch.Tensor) and embeddings.numel():
        meta_out.setdefault("embedding_dim", int(embeddings.size(1)))
    else:
        meta_out.setdefault("embedding_dim", 0)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta_out, handle, indent=2, sort_keys=True)

    print(
        f"[DUMP] Wrote partition store to {out_dir}: {written} non-empty parts "
        f"(of {len(part_ids)}), embeddings={'yes' if isinstance(embeddings, torch.Tensor) else 'no'}",
        flush=True,
    )


def run_one_model(
    label,
    model_path,
    data,
    hierarchy,
    queries,
    device,
    budgets,
    method,
    signature_name,
    solver_timeout,
    glasgow_bin,
    seed_count,
    require_node_fullcov,
    prune_query_labels,
    cache_dir,
    cache_key,
    max_component_diag_nodes,
    skip_solver,
    component_solve,
    max_component_solver_components,
    use_overlap,
    overlap_policy=None,
    label_source="feature",
    preloaded=None,
    query_workers=1,
    partial_output_prefix=None,
    partial_every=25,
    dump_partition_store_dir="",
):
    preloaded = preloaded or {}
    overlap_policy = overlap_policy or {}
    selective_overlap_active = overlap_policy_is_active(overlap_policy)
    # Node label tensor for matching/pruning/feature-index. "class" uses the real
    # node label (the intended system); "feature" uses the per-feature-vector hash.
    # Proper heterogeneous labeling: combine node TYPE with the per-node class so that
    # non-paper node types (author/institution/field, which have no class in OGBN-MAG)
    # get distinct labels per type rather than collapsing into one junk bucket. Papers
    # are split by venue; homogeneous graphs (single type) reduce to the plain class.
    node_label_full = None
    class_venue_base = None
    if label_source == "class":
        if not hasattr(data, "y") or data.y is None:
            raise ValueError("--label-source class requires node class labels data.y")
        y = data.y.detach().cpu().view(-1).long()
        node_type = getattr(data, "node_type", None)
        if node_type is not None:
            node_type = node_type.detach().cpu().view(-1).long()
        else:
            node_type = torch.zeros(data.num_nodes, dtype=torch.long)
        venue_base = (int(y[y >= 0].max()) + 1) if bool((y >= 0).any()) else 1
        class_venue_base = venue_base
        # nodes with a class (papers) -> their class; nodes without (-1) -> class 0,
        # then offset by node type so each type occupies a disjoint label band.
        within_type = torch.where(y >= 0, y, torch.zeros_like(y))
        node_label_full = node_type * venue_base + within_type
        data.node_label = node_label_full
        n_types = int(node_type.max()) + 1
        print(f"[LABEL] class-label matching: {int(node_label_full.unique().numel())} distinct "
              f"labels across {n_types} node type(s)", flush=True)
    elif str(label_source).startswith("feature_bucket_"):
        bucket_count = int(str(label_source).rsplit("_", 1)[-1])
        node_label_full = load_or_build_feature_bucket_label_tokens(
            data, cache_dir, cache_key, bucket_count
        )
        data.node_label = node_label_full
        print(
            f"[LABEL] feature-bucket matching: K={bucket_count}, "
            f"{int(node_label_full.unique().numel())} distinct labels",
            flush=True,
        )
    needs_encoder = method_needs_encoder(method)
    fullgraph_fast_path = (
        method == "all"
        and not use_overlap
        and (signature_name or "none") == "none"
        and not component_solve
        and not prune_query_labels
    )
    if not needs_encoder:
        encoder, model_load_time = None, 0.0
    elif preloaded.get("encoder") is not None:
        encoder = preloaded["encoder"]
        model_load_time = float(preloaded.get("model_load_time", 0.0))
    else:
        encoder, model_load_time = bench.load_model(model_path, data.x.size(1), device)
    adj_t = preloaded.get("adj_t")
    if adj_t is None:
        edge_values = data.edge_type.long() if hasattr(data, "edge_type") and data.edge_type is not None else None
        adj_t = SparseTensor.from_edge_index(
            data.edge_index, edge_attr=edge_values, sparse_sizes=(data.num_nodes, data.num_nodes)
        )
    if not needs_encoder:
        faiss_index, faiss_to_coarse, faiss_build_time = None, {}, 0.0
    elif preloaded.get("faiss_index") is not None:
        faiss_index = preloaded["faiss_index"]
        faiss_to_coarse = preloaded["faiss_to_coarse"]
        faiss_build_time = float(preloaded.get("faiss_build_time", 0.0))
    else:
        faiss_index, faiss_to_coarse, faiss_build_time = build_or_load_faiss_index(
            data, hierarchy, encoder, device, model_path, cache_dir, cache_key
        )
    coarse_mean_features = None
    if method in {"mean_feature", "coarse_mean_rrf"}:
        coarse_mean_features = preloaded.get("coarse_mean_features")
        if coarse_mean_features is None:
            coarse_mean_features = load_or_build_coarse_mean_features(
                data, hierarchy, cache_dir, cache_key
            )
    topo_pack = None
    if method == "topo_feature":
        topo_pack = preloaded.get("topo_pack")
        if topo_pack is None:
            topo_pack = build_coarse_topo_features(data, hierarchy, adj_t, cache_dir, cache_key)
    feature_index_pack = None
    if method == "feature_index":
        feature_index_pack = preloaded.get("feature_index_pack")
        if feature_index_pack is None:
            feature_index_pack = build_coarse_feature_index(data, hierarchy, cache_dir, cache_key, node_labels_override=node_label_full)
    tokens = None
    if signature_name and signature_name != "none":
        tokens_by_name = preloaded.get("tokens_by_name")
        if tokens_by_name is None:
            tokens_by_name = load_or_build_signature_tokens(data, cache_dir, cache_key)
        if signature_name not in tokens_by_name:
            raise ValueError(
                f"Unknown signature {signature_name}; options: {sorted(tokens_by_name)}"
            )
        tokens = tokens_by_name[signature_name]
    label_tokens = None
    if prune_query_labels or overlap_policy.get("label_compatible"):
        if node_label_full is not None:
            label_tokens = node_label_full
        else:
            label_tokens = preloaded.get("label_tokens")
            if label_tokens is None:
                label_tokens = load_or_build_feature_label_tokens(data, cache_dir, cache_key)
    if selective_overlap_active:
        load_or_build_overlap_neighbor_index(hierarchy, cache_dir, cache_key)
    if not hasattr(data, "global_id") or data.global_id is None:
        data.global_id = torch.arange(data.num_nodes, dtype=torch.long)
    fullgraph_nodes = set(range(int(data.num_nodes))) if fullgraph_fast_path else None
    fullgraph_target_name = (
        f"fullgraph_{safe_cache_key(label, cache_key, data.num_nodes, int(data.edge_index.size(1)))}"
        if fullgraph_fast_path
        else None
    )

    if dump_partition_store_dir:
        coarse_embeddings = None
        if preloaded.get("coarse_embeddings") is not None:
            coarse_embeddings = preloaded["coarse_embeddings"]
        meta = {
            "dataset": getattr(data, "_dataset_name", "") or "",
            "num_nodes": int(data.num_nodes),
            "num_coarse": len(hierarchy["coarse_part_node_sets"]),
            "signature": signature_name or "none",
            "label_source": label_source,
            "class_venue_base": class_venue_base,
            "query_pruning_source": "query_payload_v1",
            "embedding_dim": int(data.x.size(1)) if data.x is not None else 0,
        }
        dump_partition_store(
            dump_partition_store_dir,
            data,
            hierarchy,
            tokens,
            label_tokens,
            node_label_full,
            adj_t,
            faiss_index,
            faiss_to_coarse,
            coarse_embeddings,
            meta,
        )
        return []

    encoder_lock = threading.Lock() if needs_encoder else None

    def run_query(query_number, item):
        if query_number == 1 or query_number % 25 == 0:
            print(
                f"  Cascade query {query_number}/{len(queries)} start "
                f"method={method} size={item['target_query_size']} type={item.get('query_type', 'k_hop')}",
                flush=True,
            )
        query = item["query"]
        query_component_sizes = bench._subgraph_component_sizes(query)
        query_component_count = len(query_component_sizes)
        query_largest_component_nodes = query_component_sizes[0] if query_component_sizes else 0
        query_nodes = [int(node) for node in item["query_nodes"].tolist()]
        true_coarse = set(int(x) for x in item["true_coarse"])
        # Serving labels are derived from the query payload.  The planted IDs in
        # item["query_nodes"] remain available only for coverage/accuracy audits.
        q_labels = derive_query_labels(
            query,
            label_source=label_source,
            class_venue_base=class_venue_base,
        )
        query.node_label = torch.tensor(q_labels, dtype=torch.long)
        if not needs_encoder:
            model_ranking, retrieval_time = [], 0.0
        else:
            # Shared encoder/index are read-only, but PyTorch CPU inference can
            # oversubscribe badly if many threads enter the model together.
            with encoder_lock:
                model_ranking, retrieval_time = build_faiss_ranking(
                    query, encoder, device, faiss_index, faiss_to_coarse
                )
        if method == "mean_feature":
            mean_start = time.perf_counter()
            ranking = rank_by_mean_feature(query, coarse_mean_features)
            retrieval_time += time.perf_counter() - mean_start
        elif method == "coarse_mean_rrf":
            mean_start = time.perf_counter()
            mean_ranking = rank_by_mean_feature(query, coarse_mean_features)
            ranking = reciprocal_rank_fusion([model_ranking, mean_ranking])
            retrieval_time += time.perf_counter() - mean_start
        elif method == "topo_feature":
            topo_start = time.perf_counter()
            topo_features, topo_ids, topo_mean, topo_std = topo_pack
            ranking = rank_by_topo_feature(query, topo_features, topo_ids, topo_mean, topo_std)
            retrieval_time += time.perf_counter() - topo_start
        elif method == "feature_index":
            fi_start = time.perf_counter()
            ranking = rank_by_feature_index(query, feature_index_pack, query_labels=q_labels)
            retrieval_time += time.perf_counter() - fi_start
        else:
            ranking = model_ranking
        rank_pos = {int(pid): pos + 1 for pos, pid in enumerate(ranking)}
        true_ranks = [rank_pos.get(int(pid), 10**9) for pid in true_coarse]
        max_true_rank = max(true_ranks) if true_ranks else 0
        solved = False
        total_solver_time = 0.0
        total_candidate_time = 0.0
        cascade_rows = []

        for budget in budgets:
            candidate_start = time.perf_counter()
            selected = unique_ordered(
                select_ids_for_budget(
                    method,
                    ranking,
                    budget,
                    hierarchy,
                    seed_count,
                    query_id=item["query_id"],
                )
            )
            selected_set = set(selected)
            hard_missed = sorted(true_coarse - selected_set)
            true_selected = true_coarse & selected_set
            if fullgraph_fast_path:
                overlap_nodes = fullgraph_nodes
                overlap_full, overlap_missed_count = True, 0
                pruned_nodes = fullgraph_nodes
                signature_candidate_nodes = int(data.num_nodes)
                pruned_full, pruned_missed_count = True, 0
                component_count, largest_component_nodes = -1, -1
                candidate_edges = int(data.edge_index.size(1))
            else:
                overlap_nodes = selective_overlap_for_parts(
                    selected,
                    hierarchy,
                    policy={**overlap_policy, "use_overlap": use_overlap},
                    query=query,
                    label_tokens=label_tokens,
                    query_labels=q_labels,
                )
                overlap_full, overlap_missed_count = node_fullcov(query_nodes, overlap_nodes)
                pruned_nodes = prune_nodes_by_signature(
                    overlap_nodes, query, tokens, signature_name
                )
                signature_candidate_nodes = len(pruned_nodes)
                if prune_query_labels:
                    pruned_nodes = prune_nodes_by_query_label_tokens(pruned_nodes, query, label_tokens, query_labels=q_labels)
                pruned_full, pruned_missed_count = node_fullcov(query_nodes, pruned_nodes)
                if max_component_diag_nodes > 0 and len(pruned_nodes) <= max_component_diag_nodes:
                    component_count, largest_component_nodes = component_diagnostics(pruned_nodes, adj_t)
                    node_tensor = torch.tensor(sorted(int(n) for n in pruned_nodes), dtype=torch.long)
                    candidate_edges = int(adj_t[node_tensor, node_tensor].nnz()) if int(node_tensor.numel()) else 0
                else:
                    component_count, largest_component_nodes = -1, -1
                    candidate_edges = -1
            candidate_time = time.perf_counter() - candidate_start
            total_candidate_time += candidate_time
            if candidate_time > 10:
                print(
                    f"  Slow candidate q={query_number} budget={budget} "
                    f"method={method} pruned={len(pruned_nodes)} time={candidate_time:.2f}s",
                    flush=True,
                )

            solver_result_found = False
            solver_timed_out = False
            solver_time = 0.0
            target_nodes_count = 0
            component_solver_components = 0
            component_solver_nodes = 0
            if (not skip_solver) and (not require_node_fullcov or pruned_full):
                candidate_sets = [None] if fullgraph_fast_path else [sorted(pruned_nodes)]
                if (not fullgraph_fast_path) and component_solve and label_tokens is not None:
                    q_counts = query_label_counts(query, query_labels=q_labels)
                    comps = connected_components(pruned_nodes, adj_t)
                    comps = [
                        comp
                        for comp in comps
                        if component_covers_query_labels(comp, label_tokens, q_counts)
                    ]
                    comps.sort(key=len)
                    candidate_sets = comps[:max_component_solver_components] or []
                    component_solver_components = len(candidate_sets)
                    component_solver_nodes = sum(len(comp) for comp in candidate_sets)
                for candidate_nodes in candidate_sets:
                    target_name = None
                    if fullgraph_fast_path:
                        target_graph = data
                        target_global_ids = data.global_id
                        target_name = fullgraph_target_name
                        target_nodes_count += int(data.num_nodes)
                    else:
                        target_tensor = torch.tensor(candidate_nodes, dtype=torch.long)
                        target_nodes_count += int(target_tensor.numel())
                        target_graph = bench.extract_subgraph(adj_t, target_tensor, data)
                        if target_graph is None or target_graph.num_nodes <= 0:
                            continue
                        target_global_ids = target_graph.global_id
                        if node_label_full is not None:
                            target_graph.node_label = node_label_full[target_global_ids.long()]
                    solve_start = time.perf_counter()
                    solver_result = glasgow_solve(
                        query_data=query,
                        target_data=target_graph,
                        query_global_ids=item["query_nodes"],
                        target_global_ids=target_global_ids,
                        max_solutions=1,
                        timeout_seconds=solver_timeout,
                        binary_path=glasgow_bin,
                        target_name=target_name,
                    )
                    elapsed = time.perf_counter() - solve_start
                    solver_time += elapsed
                    total_solver_time += elapsed
                    if elapsed > 10:
                        print(
                            f"  Slow solver q={query_number} budget={budget} "
                            f"method={method} nodes={target_nodes_count} time={elapsed:.2f}s",
                            flush=True,
                        )
                    solver_result_found = bool(solver_result.found)
                    solver_timed_out = bool(solver_result.timed_out)
                    if solver_result_found or solver_timed_out:
                        break

            row = {
                "query_number": query_number,
                "model": label,
                "model_path": model_path,
                "query_id": item["query_id"],
                "query_type": item.get("query_type", "k_hop"),
                "is_negative": bool(item.get("is_negative", False)),
                "negative_type": item.get("negative_type", ""),
                "expected_match": bool(item.get("expected_match", True)),
                "target_query_size": item["target_query_size"],
                "query_nodes": query.num_nodes,
                "query_pruning_source": "query_payload_v1",
                "query_component_count": query_component_count,
                "query_largest_component_nodes": query_largest_component_nodes,
                "full_graph_nodes": data.num_nodes,
                "true_coarse_count": len(true_coarse),
                "budget": budget,
                "method": method,
                "signature": signature_name or "none",
                "hard_coarse_fullcov": bool(true_coarse) and not hard_missed,
                "hard_missed_coarse_count": len(hard_missed),
                "selected_true_coarse_count": len(true_selected),
                "coarse_precision_at_budget": (len(true_selected) / len(selected)) if selected else 0.0,
                "coarse_recall_at_budget": (len(true_selected) / len(true_coarse)) if true_coarse else 0.0,
                "coarse_hit_at_budget": bool(true_selected),
                "max_true_coarse_rank": max_true_rank,
                "impossible_at_budget": len(true_coarse) > int(budget),
                "use_overlap": use_overlap,
                "overlap_max_parts": overlap_policy.get("max_parts") if overlap_policy.get("max_parts") is not None else -1,
                "overlap_min_support": overlap_policy.get("min_support") if overlap_policy.get("min_support") is not None else -1,
                "overlap_max_nodes": overlap_policy.get("max_nodes") if overlap_policy.get("max_nodes") is not None else -1,
                "overlap_label_compatible": bool(overlap_policy.get("label_compatible", False)),
                "overlap_bridge_infill_top": overlap_policy.get("bridge_infill_top") if overlap_policy.get("bridge_infill_top") is not None else -1,
                "overlap_node_fullcov": overlap_full,
                "overlap_missed_node_count": overlap_missed_count,
                "overlap_candidate_nodes": len(overlap_nodes),
                "pruned_node_fullcov": pruned_full,
                "pruned_missed_node_count": pruned_missed_count,
                "signature_candidate_nodes": signature_candidate_nodes,
                "query_label_pruning": prune_query_labels,
                "pruned_candidate_nodes": len(pruned_nodes),
                "full_graph_edges": int(data.edge_index.size(1)),
                "candidate_edges_diag": candidate_edges,
                "edge_reduction_factor": (float(data.edge_index.size(1)) / candidate_edges) if candidate_edges and candidate_edges > 0 else 0.0,
                "candidate_node_fraction": (len(pruned_nodes) / data.num_nodes) if data.num_nodes else 0.0,
                "node_reduction_factor": (data.num_nodes / len(pruned_nodes)) if pruned_nodes else 0.0,
                "pruned_component_count": component_count,
                "pruned_largest_component_nodes": largest_component_nodes,
                "target_nodes": target_nodes_count,
                "component_solver": component_solve,
                "component_solver_components": component_solver_components,
                "component_solver_nodes": component_solver_nodes,
                "retrieval_time_seconds": retrieval_time,
                "candidate_time_seconds": candidate_time,
                "solver_time_seconds": solver_time,
                "solver_found": solver_result_found,
                "false_positive": bool(item.get("is_negative", False)) and solver_result_found,
                "solver_timed_out": solver_timed_out,
                "skipped_by_node_fullcov_guard": require_node_fullcov and not pruned_full,
                "model_load_time_seconds": model_load_time,
                "faiss_build_time_seconds": faiss_build_time,
            }
            cascade_rows.append(row)
            if solver_result_found:
                solved = True
                row["cascade_first_solved"] = True
                row["cascade_solved_budget"] = budget
                break
            if solver_timed_out:
                row["cascade_first_solved"] = False
                row["cascade_solved_budget"] = 0
                break

        if not solved:
            for row in cascade_rows:
                row.setdefault("cascade_first_solved", False)
                row.setdefault("cascade_solved_budget", 0)
        for row in cascade_rows:
            row["cascade_total_solver_time_seconds"] = total_solver_time
            row["cascade_total_candidate_time_seconds"] = total_candidate_time

        if query_number % 10 == 0 or query_number == len(queries):
            solved_count = sum(
                1
                for row in cascade_rows
                if row.get("cascade_first_solved") and int(row["budget"]) in budgets
            )
            print(
                f"  Cascade query {query_number}/{len(queries)} complete; "
                f"solved so far: {solved_count}",
                flush=True,
            )

        return cascade_rows

    rows = []
    query_workers = max(1, int(query_workers or 1))
    partial_every = max(0, int(partial_every or 0))
    completed_query_numbers = set()
    partial_path = f"{partial_output_prefix}_partial_per_query.csv" if partial_output_prefix else None
    if partial_path and os.path.exists(partial_path):
        try:
            with open(partial_path, newline="", encoding="utf-8") as handle:
                prior_rows = list(csv.DictReader(handle))
            rows.extend(prior_rows)
            completed_query_numbers = {
                int(row["query_number"])
                for row in prior_rows
                if str(row.get("query_number", "")).isdigit()
            }
            if completed_query_numbers:
                print(
                    f"  Resuming {label}: loaded {len(prior_rows)} rows "
                    f"for {len(completed_query_numbers)} completed queries",
                    flush=True,
                )
        except Exception as exc:
            print(f"  Could not load partial rows from {partial_path}: {exc}", flush=True)
    if query_workers == 1:
        for query_number, item in enumerate(queries, start=1):
            if query_number in completed_query_numbers:
                continue
            rows.extend(run_query(query_number, item))
            if partial_output_prefix and partial_every and query_number % partial_every == 0:
                write_csv(f"{partial_output_prefix}_partial_per_query.csv", rows)
    else:
        print(f"  Query-level workers={query_workers} method={method} model={label}", flush=True)
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=query_workers) as pool:
            futures = {
                pool.submit(run_query, query_number, item): query_number
                for query_number, item in enumerate(queries, start=1)
                if query_number not in completed_query_numbers
            }
            for fut in concurrent.futures.as_completed(futures):
                query_number = futures[fut]
                query_rows = fut.result()
                rows.extend(query_rows)
                completed += 1
                if completed % 10 == 0 or completed == len(queries):
                    solved_count = sum(
                        1
                        for row in rows
                        if row.get("cascade_first_solved") and int(row["budget"]) in budgets
                    )
                    print(
                        f"  Cascade queries completed {completed}/{len(queries)} "
                        f"method={method} solved={solved_count}",
                        flush=True,
                    )
                if partial_output_prefix and partial_every and completed % partial_every == 0:
                    write_csv(f"{partial_output_prefix}_partial_per_query.csv", rows)
    rows.sort(key=lambda row: (int(row.get("query_number", 0)), int(row.get("budget", 0))))
    if partial_output_prefix:
        write_csv(f"{partial_output_prefix}_partial_per_query.csv", rows)
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["model"], row.get("query_type", "k_hop"), row["target_query_size"], row["method"], row["signature"])
        groups.setdefault(key, []).append(row)
    summaries = []
    def parse_bool(value):
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    budgets = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    def parse_budget(value):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    def budget_columns(solved_budgets):
        out = {}
        running_solved = 0
        for budget in budgets:
            exact = sum(1 for b in solved_budgets.values() if b == budget)
            running_solved += exact
            out[f"solved_at_{budget}"] = exact
            out[f"first_solved_at_{budget}"] = exact
            out[f"solved_by_{budget}"] = running_solved
        return out

    for (model, query_type, target_size, method, signature), group in groups.items():
        # Count one row per query for cascade outcomes.
        query_ids = sorted({row["query_id"] for row in group})
        solved_budgets = {}
        expected_match = {}
        timed_out_by_query = {}
        for qid in query_ids:
            qrows = [row for row in group if row["query_id"] == qid]
            solved = [row for row in qrows if row.get("cascade_first_solved")]
            solved_budget_values = [parse_budget(row.get("budget")) for row in solved]
            solved_budget_values = [budget for budget in solved_budget_values if budget > 0]
            solved_budgets[qid] = min(solved_budget_values) if solved_budget_values else 0
            expected_match[qid] = parse_bool(qrows[0].get("expected_match", True))
            timed_out_by_query[qid] = any(parse_bool(row.get("solver_timed_out")) for row in qrows)
        positive_ids = [qid for qid in query_ids if expected_match.get(qid, True)]
        negative_ids = [qid for qid in query_ids if not expected_match.get(qid, True)]
        summaries.append(
            {
                "model": model,
                "query_type": query_type,
                "target_query_size": target_size,
                "method": method,
                "signature": signature,
                "queries": len(query_ids),
                "solved_total": sum(1 for b in solved_budgets.values() if b > 0),
                "positive_queries": len(positive_ids),
                "negative_queries": len(negative_ids),
                "positive_solved": sum(1 for qid in positive_ids if solved_budgets.get(qid, 0) > 0),
                "false_positives": sum(1 for qid in negative_ids if solved_budgets.get(qid, 0) > 0),
                "correct_no_match": sum(
                    1
                    for qid in negative_ids
                    if solved_budgets.get(qid, 0) == 0 and not timed_out_by_query.get(qid, False)
                ),
                "negative_timeouts": sum(1 for qid in negative_ids if timed_out_by_query.get(qid, False)),
                "unknown_within_budget": sum(
                    1
                    for qid in query_ids
                    if solved_budgets.get(qid, 0) == 0 and timed_out_by_query.get(qid, False)
                ),
                **budget_columns(solved_budgets),
                "unsolved": sum(1 for b in solved_budgets.values() if b == 0),
                "avg_solver_time_per_query": mean(
                    [
                        max(
                            float(row["cascade_total_solver_time_seconds"])
                            for row in group
                            if row["query_id"] == qid
                        )
                        for qid in query_ids
                    ]
                ),
                "avg_candidate_time_per_query": mean(
                    [
                        max(
                            float(row["cascade_total_candidate_time_seconds"])
                            for row in group
                            if row["query_id"] == qid
                        )
                        for qid in query_ids
                    ]
                ),
                "avg_retrieval_time": mean(
                    [float(row["retrieval_time_seconds"]) for row in group]
                ),
                "avg_total_time_per_query": mean(
                    [
                        max(
                            float(row["cascade_total_solver_time_seconds"])
                            + float(row["cascade_total_candidate_time_seconds"])
                            for row in group
                            if row["query_id"] == qid
                        )
                        for qid in query_ids
                    ]
                ),
                "avg_pruned_nodes": mean(
                    [float(row["pruned_candidate_nodes"]) for row in group]
                ),
                "avg_signature_nodes": mean(
                    [float(row["signature_candidate_nodes"]) for row in group]
                ),
                "avg_overlap_nodes": mean(
                    [float(row["overlap_candidate_nodes"]) for row in group]
                ),
                "avg_node_reduction_factor": mean(
                    [float(row["node_reduction_factor"]) for row in group]
                ),
                "avg_edge_reduction_factor_diag": mean(
                    [
                        float(row["edge_reduction_factor"])
                        for row in group
                        if float(row.get("edge_reduction_factor", 0) or 0) > 0
                    ]
                ),
                "avg_candidate_node_fraction": mean(
                    [float(row["candidate_node_fraction"]) for row in group]
                ),
                "avg_precision_at_budget": mean(
                    [float(row["coarse_precision_at_budget"]) for row in group]
                ),
                "avg_recall_at_budget": mean(
                    [float(row["coarse_recall_at_budget"]) for row in group]
                ),
                "hit_rows": sum(parse_bool(row["coarse_hit_at_budget"]) for row in group),
                "avg_max_true_coarse_rank": mean(
                    [float(row["max_true_coarse_rank"]) for row in group]
                ),
                "impossible_rows": sum(parse_bool(row["impossible_at_budget"]) for row in group),
                "pruned_fullcov_rows": sum(
                    parse_bool(row["pruned_node_fullcov"]) for row in group
                ),
                "solver_timeouts": sum(parse_bool(row["solver_timed_out"]) for row in group),
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mag")
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--target-sizes", default="20")
    parser.add_argument("--query-types", default="k_hop")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--data-root", default="/data/datasets")
    parser.add_argument("--hierarchy-path", default="")
    parser.add_argument("--model", action="append", default=[], help="label=path")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--budgets", default="20,50,100")
    parser.add_argument("--method", choices=["fixed", "hybrid", "all", "random", "mean_feature", "coarse_mean_rrf", "topo_feature", "feature_index"], default="fixed")
    parser.add_argument("--signature", default="type_rel_feat32")
    parser.add_argument("--solver-timeout", type=float, default=5.0)
    parser.add_argument("--glasgow-bin", default=os.environ.get("GLASGOW_SOLVER_BIN", "/usr/local/bin/glasgow_subgraph_solver"))
    parser.add_argument("--stitch-seed-count", type=int, default=20)
    parser.add_argument("--require-node-fullcov", action="store_true")
    parser.add_argument("--prune-query-labels", action="store_true")
    parser.add_argument("--cache-dir", default="/cache/overlap_cascade")
    parser.add_argument("--max-component-diag-nodes", type=int, default=50000)
    parser.add_argument("--skip-solver", action="store_true")
    parser.add_argument("--component-solve", action="store_true")
    parser.add_argument("--max-component-solver-components", type=int, default=20)
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--overlap-max-parts", type=int, default=0, help="selective overlap: per selected partition, keep only its top-N neighbor partitions by boundary support (0=off)")
    parser.add_argument("--overlap-min-support", type=int, default=0, help="selective overlap: drop neighbor partitions contributing fewer than N boundary nodes (0=off)")
    parser.add_argument("--overlap-max-nodes", type=int, default=0, help="selective overlap: global cap on overlap nodes added, greedy by neighbor support (0=off)")
    parser.add_argument("--overlap-label-compatible", action="store_true", help="selective overlap: only add overlap nodes whose label token matches a query label")
    parser.add_argument("--overlap-bridge-infill-top", type=int, default=0, help="recall-expansion: add full nodes of the top-N partitions most strongly bridged (by boundary support) to the selected set (targets dispersed/random-walk queries; 0=off, 8 recommended)")
    parser.add_argument("--overlap-bridge-infill-min-support", type=int, default=0, help="minimum total boundary support for a bridge-infill candidate")
    parser.add_argument("--no-boundary-overlap", action="store_true", help="skip one-hop boundary overlap (for bridge-infill-only experiments)")
    parser.add_argument("--label-source", default="feature", help="node label for matching/pruning/feature-index: 'feature'=per-feature-vector hash, 'class'=real class label data.y, 'feature_bucket_K'=MD5(feature tuple) modulo K")
    parser.add_argument("--generate-query-cache-only", action="store_true")
    parser.add_argument(
        "--evaluation-query-types",
        default="",
        help=(
            "Optional query-family subset to evaluate after loading the cache. "
            "The cache is still generated and validated from --query-types, so a "
            "canonical all-family cache can drive a targeted rerun without changing "
            "query identities."
        ),
    )
    parser.add_argument("--max-eval-queries", type=int, default=0)
    parser.add_argument("--query-workers", type=int, default=1)
    parser.add_argument("--partial-every", type=int, default=25)
    parser.add_argument("--dump-partition-store", default="", help="if set, dump a per-partition on-disk store to this DIR (for streaming/out-of-core serving) and skip the query loop")
    args = parser.parse_args()

    budgets = parse_budgets(args.budgets)
    overlap_policy = {
        "max_parts": args.overlap_max_parts if args.overlap_max_parts > 0 else None,
        "min_support": args.overlap_min_support if args.overlap_min_support > 0 else None,
        "max_nodes": args.overlap_max_nodes if args.overlap_max_nodes > 0 else None,
        "label_compatible": bool(args.overlap_label_compatible),
        "bridge_infill_top": args.overlap_bridge_infill_top if args.overlap_bridge_infill_top > 0 else None,
        "bridge_infill_min_support": args.overlap_bridge_infill_min_support,
        "boundary_overlap": not args.no_boundary_overlap,
    }
    if overlap_policy_is_active(overlap_policy):
        print(f"Selective overlap policy: {overlap_policy}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_named_data(args.dataset, args.data_root)
    hierarchy_path = args.hierarchy_path or default_hierarchy_path(args.dataset)
    cache_key = safe_cache_key(
        args.dataset,
        path_fingerprint(hierarchy_path),
        data.num_nodes,
        data.edge_index.size(1),
    )
    hierarchy = load_or_prepare_hierarchy(
        data, hierarchy_path, args.cache_dir, cache_key, args.dataset
    )
    hierarchy = load_or_build_overlap_index(data, hierarchy, args.cache_dir, cache_key)

    queries = load_or_generate_cascade_queries(
        data,
        hierarchy,
        args.queries,
        args.target_sizes,
        args.seed,
        args.query_types,
        args.cache_dir,
        cache_key,
    )
    if args.generate_query_cache_only:
        print(f"Query cache ready for {len(queries)} queries.", flush=True)
        return
    if args.evaluation_query_types:
        selected_types = set(_expanded_query_types(args.evaluation_query_types))
        unknown_types = selected_types - set(_expanded_query_types(args.query_types))
        if unknown_types:
            raise ValueError(
                "--evaluation-query-types must be a subset of --query-types; "
                f"unknown={sorted(unknown_types)}"
            )
        original = len(queries)
        queries = [
            item for item in queries if str(item.get("query_type", "")) in selected_types
        ]
        expected = args.queries * len(parse_budgets(args.target_sizes)) * len(selected_types)
        if len(queries) != expected:
            raise ValueError(
                "Filtered query count does not match the canonical workload: "
                f"got={len(queries)} expected={expected} types={sorted(selected_types)}"
            )
        print(
            f"[EVALUATION SUBSET] Using {len(queries)}/{original} cached queries "
            f"for types={sorted(selected_types)}.",
            flush=True,
        )
    if args.max_eval_queries and args.max_eval_queries > 0:
        original = len(queries)
        queries = queries[: args.max_eval_queries]
        print(f"[SMOKE] Using first {len(queries)}/{original} cached queries.", flush=True)

    if not args.model:
        if method_needs_encoder(args.method):
            raise ValueError("--model label=path is required for encoder-ranked methods")
        args.model = [f"{args.dataset}="]

    all_rows = []
    for spec in args.model:
        label, path = spec.split("=", 1)
        all_rows.extend(
            run_one_model(
                label,
                path,
                data,
                hierarchy,
                queries,
                device,
                budgets,
                args.method,
                args.signature,
                args.solver_timeout,
                args.glasgow_bin,
                args.stitch_seed_count,
                args.require_node_fullcov,
                args.prune_query_labels,
                args.cache_dir,
                cache_key,
                args.max_component_diag_nodes,
                args.skip_solver,
                args.component_solve,
                args.max_component_solver_components,
                not args.no_overlap,
                overlap_policy=overlap_policy,
                label_source=args.label_source,
                query_workers=args.query_workers,
                partial_output_prefix=f"{args.output_prefix}_{clean_tag(label)}",
                partial_every=args.partial_every,
                dump_partition_store_dir=args.dump_partition_store,
            )
        )
        if args.dump_partition_store:
            print(f"Partition store dumped to {args.dump_partition_store}; skipping query evaluation.", flush=True)
            return

    per_query_path = f"{args.output_prefix}_per_query.csv"
    summary_path = f"{args.output_prefix}_summary.csv"
    write_csv(per_query_path, all_rows)
    summary_rows = summarize(all_rows)
    write_csv(summary_path, summary_rows)
    print(f"Per-query results: {per_query_path}", flush=True)
    print(f"Summary results: {summary_path}", flush=True)
    for row in summary_rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
