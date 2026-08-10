"""Streaming / out-of-core Jigsaw subgraph-matching smoke test.

Demonstrates BOUNDED peak memory: instead of resident-holding the whole graph,
the streaming serve loads only a small on-disk partition store (produced by
``benchmark_overlap_glasgow_cascade.py --dump-partition-store DIR``) plus the
encoder and query cache, and streams per-partition records on demand with an LRU
cache capped at ``--cache-partitions`` records.

Two modes:
  --mode streaming              : the bounded-memory serve (does NOT load the full graph)
  --mode whole-graph-baseline   : loads the full graph the way the cascade does and
                                  records the resident footprint, for comparison.

Reuses the cascade's retrieval + pruning functions and the Glasgow solver.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import torch


# --- robust repo-root import path --------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_geometric.data import Data

# Cascade helpers (retrieval + pruning + model/data loaders are reused verbatim).
import benchmark_glasgow as bench
from benchmark_retrieval import default_hierarchy_path, load_named_data
from scripts.benchmark_overlap_glasgow_cascade import (
    build_faiss_ranking,
    make_l2_index,
    parse_budgets,
    prune_nodes_by_query_label_tokens,
    prune_nodes_by_signature,
    query_label_counts,
    component_covers_query_labels,
    derive_query_labels,
    load_or_prepare_hierarchy,
    load_or_build_overlap_index,
    safe_cache_key,
    path_fingerprint,
    torch_load_any,
)
from src.glasgow_solver import glasgow_solve

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

if os.name != "nt":
    import resource
else:
    resource = None


# --- RSS peak sampler ---------------------------------------------------------
class RSSPeakSampler:
    """Background thread polling process RSS; records the peak in MB."""

    def __init__(self, interval_seconds=0.05):
        self.interval = float(interval_seconds)
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process(os.getpid()) if psutil is not None else None

    def _sample_bytes(self):
        if self._proc is not None:
            try:
                return int(self._proc.memory_info().rss)
            except Exception:
                return 0
        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is KB on Linux, bytes on macOS.
            if sys.platform == "darwin":
                return int(usage)
            return int(usage) * 1024
        return 0

    def _run(self):
        while not self._stop.is_set():
            current = self._sample_bytes()
            if current > self._peak_bytes:
                self._peak_bytes = current
            self._stop.wait(self.interval)

    def start(self):
        # seed with an immediate sample so a short window still records something
        self._peak_bytes = max(self._peak_bytes, self._sample_bytes())
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._peak_bytes = max(self._peak_bytes, self._sample_bytes())
        return self.peak_mb()

    def peak_mb(self):
        return self._peak_bytes / (1024.0 * 1024.0)


# --- on-disk partition store --------------------------------------------------
class PartitionStore:
    """LRU-cached reader over ``DIR/parts/{p}.pt`` records.

    Each record is {"nodes": LongTensor[global ids], "edge_index": LongTensor[2,E]
    of GLOBAL-id endpoints}. Holds at most ``max_cached`` records resident.
    """

    def __init__(self, store_dir, max_cached=8):
        self.parts_dir = os.path.join(store_dir, "parts")
        self.max_cached = max(1, int(max_cached))
        self._cache = OrderedDict()

    def has(self, part_id):
        return os.path.exists(os.path.join(self.parts_dir, f"{int(part_id)}.pt"))

    def get(self, part_id):
        part_id = int(part_id)
        if part_id in self._cache:
            self._cache.move_to_end(part_id)
            return self._cache[part_id]
        path = os.path.join(self.parts_dir, f"{part_id}.pt")
        if not os.path.exists(path):
            record = {
                "nodes": torch.zeros(0, dtype=torch.long),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
            }
        else:
            record = torch_load_any(path)
        self._cache[part_id] = record
        self._cache.move_to_end(part_id)
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return record


def load_store_artifacts(store_dir):
    """Load the store-wide artifacts (no full graph) and rebuild the faiss index."""
    with open(os.path.join(store_dir, "meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    sig = torch_load_any(os.path.join(store_dir, "signature_tokens.pt"))
    tokens = sig.get("tokens") if isinstance(sig, dict) else sig
    if tokens is not None:
        tokens = tokens.long()

    lab = torch_load_any(os.path.join(store_dir, "label_tokens.pt"))
    label_tokens = lab.get("labels") if isinstance(lab, dict) else lab
    if label_tokens is not None:
        label_tokens = label_tokens.long()

    emb_payload = torch_load_any(os.path.join(store_dir, "coarse_embeddings.pt"))
    embeddings = emb_payload.get("embeddings")
    faiss_to_coarse = {int(k): int(v) for k, v in emb_payload.get("faiss_to_coarse", {}).items()}
    faiss_index = None
    if isinstance(embeddings, torch.Tensor) and embeddings.numel():
        faiss_index = make_l2_index(int(embeddings.size(1)))
        faiss_index.add(embeddings.detach().cpu().float().numpy())

    return meta, tokens, label_tokens, faiss_index, faiss_to_coarse


def load_encoder(model_path, in_features, device):
    """Load the encoder exactly like the cascade does (bench.load_model)."""
    encoder, _load_time = bench.load_model(model_path, in_features, device)
    return encoder


def local_connected_components(nodes_set, edges):
    """Connected components over ``nodes_set`` induced by ``edges`` (list of (u,v)
    GLOBAL-id pairs). Union-find; returns a list of lists of global node ids
    (isolated survivors become singletons). Does NOT need the full adjacency."""
    parent = {int(n): int(n) for n in nodes_set}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for u, v in edges:
        if u in parent and v in parent:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    groups = {}
    for n in list(parent.keys()):
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


def _build_target_from_nodes(pruned_nodes, edge_map, label_tokens):
    """Build a PyG target Data from pruned GLOBAL node ids and an accumulated
    global-endpoint edge set. Reindexes nodes to 0..k-1, keeps global_id, and
    attaches node_label from label_tokens when available.
    """
    pruned_sorted = sorted(int(n) for n in pruned_nodes)
    if not pruned_sorted:
        return None, torch.zeros(0, dtype=torch.long)
    global_to_local = {gid: i for i, gid in enumerate(pruned_sorted)}
    pruned_set = set(pruned_sorted)
    src_local = []
    dst_local = []
    for (u, v) in edge_map:
        if u in pruned_set and v in pruned_set:
            src_local.append(global_to_local[u])
            dst_local.append(global_to_local[v])
    if src_local:
        edge_index = torch.tensor([src_local, dst_local], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    global_id = torch.tensor(pruned_sorted, dtype=torch.long)
    # Streaming has no full feature matrix resident; targets carry a placeholder x
    # so num_nodes is well-defined. Glasgow matching uses node_label (below).
    x = torch.zeros((len(pruned_sorted), 1), dtype=torch.float32)
    target = Data(x=x, edge_index=edge_index, num_nodes=len(pruned_sorted))
    target.global_id = global_id
    if label_tokens is not None:
        target.node_label = label_tokens[global_id].long()
    return target, global_id


def run_streaming(args):
    device = torch.device("cpu")
    sampler = RSSPeakSampler().start()

    label, model_path = args.model.split("=", 1)

    meta, tokens, label_tokens, faiss_index, faiss_to_coarse = load_store_artifacts(args.store)
    if faiss_index is None:
        raise RuntimeError(
            "coarse_embeddings.pt has no embeddings; streaming retrieval needs the "
            "rebuilt faiss index. Re-dump the store with an encoder-ranked method."
        )

    queries = torch_load_any(args.queries_cache)
    if args.max_queries and args.max_queries > 0:
        queries = queries[: args.max_queries]
    budgets = parse_budgets(args.budgets)
    if not queries:
        raise RuntimeError("empty query cache")

    # Encoder input dim = the RAW node-feature dimension, not meta['embedding_dim']
    # (which is the OUTPUT/coarse-embedding dim). Queries are subgraphs of the data
    # graph, so query.x carries the same feature dimension the encoder was trained on.
    in_features = int(queries[0]["query"].x.size(1))
    encoder = load_encoder(model_path, in_features, device)

    store = PartitionStore(args.store, max_cached=args.cache_partitions)

    rows = []
    for q_number, item in enumerate(queries, start=1):
        query = item["query"]
        query_nodes = [int(n) for n in item["query_nodes"].tolist()]
        # Derive query labels from the request payload; planted IDs are metrics only.
        q_labels = derive_query_labels(
            query,
            meta.get("label_source", "feature"),
            class_venue_base=meta.get("class_venue_base"),
        )
        query.node_label = torch.tensor(q_labels, dtype=torch.long)

        ranking, retrieval_time = build_faiss_ranking(
            query, encoder, device, faiss_index, faiss_to_coarse
        )

        solved = False
        first_budget = True
        for budget in budgets:
            selected = ranking[:budget]
            # PASS 1: accumulate node IDs (vectorized; keep only node tensors, NO edges,
            # so peak memory tracks the node set, not the overlap-union edge set).
            _t = time.perf_counter()
            _node_chunks = []
            part_node_map = {}  # pid -> node ids (cheap; ~ids only), used to skip empty parts in pass 2
            for pid in selected:
                record = store.get(pid)
                nodes = record.get("nodes")
                if nodes is not None and nodes.numel():
                    _n = nodes.reshape(-1).long()
                    part_node_map[pid] = _n
                    _node_chunks.append(_n)
            acc_tensor = torch.unique(torch.cat(_node_chunks)) if _node_chunks else torch.empty(0, dtype=torch.long)
            accumulated_nodes = set(acc_tensor.tolist())
            accumulated_count = len(accumulated_nodes)
            fetch1_s = time.perf_counter() - _t
            # node-level pruning (global coverage) -- reuses the cascade operators
            _t = time.perf_counter()
            pruned_nodes = prune_nodes_by_signature(
                accumulated_nodes,
                query,
                tokens,
                meta.get("signature", "none"),
            )
            pruned_nodes = prune_nodes_by_query_label_tokens(
                pruned_nodes, query, label_tokens, query_labels=q_labels
            )
            pruned_set = set(int(n) for n in pruned_nodes)
            pruned_count = len(pruned_set)
            prune_s = time.perf_counter() - _t
            # PASS 2: fetch induced edges ONLY among survivors (re-streams records to keep
            # memory bounded to <=cache_partitions; edge filter vectorized with torch.isin,
            # one partition tensor at a time so peak memory stays the survivor subgraph).
            _t = time.perf_counter()
            pruned_tensor = (
                torch.tensor(sorted(pruned_set), dtype=torch.long)
                if pruned_set else torch.empty(0, dtype=torch.long)
            )
            _edge_chunks = []
            _parts_read = 0
            for pid in selected:
                # SKIP-EMPTY: a partition's edges are induced among its own node set, so if none of
                # its nodes survived pruning it cannot contribute any survivor edge -> skip the disk
                # re-read entirely (uses the pass-1 node ids; no extra I/O, memory bound unchanged).
                pn = part_node_map.get(pid)
                if pn is None or pruned_tensor.numel() == 0:
                    continue
                if not bool(torch.isin(pn, pruned_tensor).any()):
                    continue
                record = store.get(pid)
                _parts_read += 1
                edge_index = record.get("edge_index")
                if edge_index is None or not edge_index.numel():
                    continue
                ei = edge_index.long()
                keep = torch.isin(ei[0], pruned_tensor) & torch.isin(ei[1], pruned_tensor) & (ei[0] != ei[1])
                if bool(keep.any()):
                    _edge_chunks.append(ei[:, keep])
            if _edge_chunks:
                _surv = torch.cat(_edge_chunks, dim=1)
                survivor_edges = list(zip(_surv[0].tolist(), _surv[1].tolist()))
            else:
                survivor_edges = []
            fetch2_s = time.perf_counter() - _t
            # component-solve: decompose survivors into connected components that cover
            # the query labels, solve smallest-first (mirrors the cascade's component_solve).
            _t = time.perf_counter()
            q_counts = query_label_counts(query, query_labels=q_labels)
            comps = local_connected_components(pruned_set, survivor_edges)
            comps = [c for c in comps if component_covers_query_labels(sorted(c), label_tokens, q_counts)]
            comps.sort(key=len)
            comps = comps[: args.max_component_solver_components]
            compbuild_s = time.perf_counter() - _t
            solver_found = False
            solver_time = 0.0
            solver_nodes = 0
            for comp in comps:
                target, global_id = _build_target_from_nodes(comp, survivor_edges, label_tokens)
                if target is None or target.num_nodes < query.num_nodes:
                    continue
                solver_nodes += int(target.num_nodes)
                solve_start = time.perf_counter()
                solver_result = glasgow_solve(
                    query_data=query,
                    target_data=target,
                    query_global_ids=item["query_nodes"],
                    target_global_ids=global_id,
                    max_solutions=1,
                    timeout_seconds=args.solver_timeout,
                    binary_path=args.glasgow_bin,
                )
                solver_time += time.perf_counter() - solve_start
                if bool(solver_result.found):
                    solver_found = True
                    break

            # retrieval is done once per query; attribute it to the first budget only
            retrieval_this = retrieval_time if first_budget else 0.0
            budget_total_s = (
                retrieval_this + fetch1_s + prune_s + fetch2_s + compbuild_s + solver_time
            )
            rows.append(
                {
                    "query_number": q_number,
                    "query_id": item.get("query_id", ""),
                    "query_type": item.get("query_type", ""),
                    "target_query_size": item.get("target_query_size", 0),
                    "query_nodes": query.num_nodes,
                    "query_pruning_source": "query_payload_v1",
                    "budget": budget,
                    "retrieved_parts": len(selected),
                    "parts_read_pass2": _parts_read,
                    "accumulated_nodes": accumulated_count,
                    "pruned_nodes": pruned_count,
                    "components_covering": len(comps),
                    "component_solver_nodes": solver_nodes,
                    "found": solver_found,
                    "retrieval_time_seconds": round(retrieval_this, 6),
                    "fetch1_s": round(fetch1_s, 6),
                    "prune_s": round(prune_s, 6),
                    "fetch2_s": round(fetch2_s, 6),
                    "compbuild_s": round(compbuild_s, 6),
                    "solver_time_seconds": solver_time,
                    "budget_total_s": round(budget_total_s, 6),
                    "peak_rss_mb": round(sampler.peak_mb(), 2),
                }
            )
            first_budget = False
            if solver_found:
                solved = True
                break

        # incremental flush so partial results survive interruption on long runs
        _write_csv(f"{args.output_prefix}_streaming_per_query.csv", rows)
        print(
            f"[STREAM] query {q_number}/{len(queries)} solved={solved} peak_rss={sampler.peak_mb():.1f}MB",
            flush=True,
        )

    peak = sampler.stop()
    out_path = f"{args.output_prefix}_streaming_per_query.csv"
    _write_csv(out_path, rows)
    print(f"[STREAM] wrote {out_path}; peak RSS {peak:.1f} MB over {len(queries)} queries", flush=True)


def run_whole_graph_baseline(args):
    """Load the full graph + hierarchy the way the cascade does; record peak RSS."""
    sampler = RSSPeakSampler().start()

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

    num_nodes = int(data.num_nodes)
    num_edges = int(data.edge_index.size(1))
    peak = sampler.stop()

    rows = [
        {
            "dataset": args.dataset,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "peak_rss_mb": round(peak, 2),
        }
    ]
    out_path = f"{args.output_prefix}_baseline_rss.csv"
    _write_csv(out_path, rows)
    print(
        f"[BASELINE] {args.dataset}: {num_nodes:,} nodes, {num_edges:,} edges, "
        f"peak RSS {peak:.1f} MB -> {out_path}",
        flush=True,
    )


def _write_csv(path, rows):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="", help="partition store DIR from --dump-partition-store")
    parser.add_argument("--model", default="", help="label=path (encoder loaded like the cascade)")
    parser.add_argument("--dataset", default="mag")
    parser.add_argument("--queries-cache", default="", help="path to a *_queries.pt cache")
    parser.add_argument("--budgets", default="20,50,100,200,500,1000")
    parser.add_argument("--cache-partitions", type=int, default=8, help="max part records held resident at once (LRU)")
    parser.add_argument("--solver-timeout", type=float, default=120.0)
    parser.add_argument("--glasgow-bin", default=os.environ.get("GLASGOW_SOLVER_BIN", "/usr/local/bin/glasgow_subgraph_solver"))
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--mode", choices=["streaming", "whole-graph-baseline"], default="streaming")
    parser.add_argument("--data-root", default="/data/datasets")
    parser.add_argument("--hierarchy-path", default="")
    parser.add_argument("--cache-dir", default="/cache/overlap_cascade")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-component-solver-components", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "streaming":
        if not args.store or not args.model or not args.queries_cache:
            raise ValueError("--mode streaming requires --store, --model, and --queries-cache")
        run_streaming(args)
    else:
        run_whole_graph_baseline(args)


if __name__ == "__main__":
    main()
