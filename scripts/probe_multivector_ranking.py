"""Phase-0 multi-vector query-embedding probe (inference-only, no Glasgow).

Compares single-vector query ranking (current retriever) against multi-vector
(subquery-decomposition + MaxSim) ranking, on the metric where the single-vector
retriever fails: max_true_coarse_rank / FullCov@K, per query family.

Reuses the cascade's encoder, hierarchy, query generation, and partition-embedding
build so results are directly comparable to runs/label_selectivity_experiments/.
Writes a per-query CSV; analysis is done offline.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_glasgow as bench
from src.model import get_graph_embedding
import benchmark_overlap_glasgow_cascade as C  # module namespace: exposes both defined + imported names


def partition_embed_matrix(data, hierarchy, encoder, device, model_path, cache_dir, cache_key):
    """Return (P [num_parts x d] L2-normalized, coarse_ids list) reusing the cascade cache."""
    C.build_or_load_faiss_index(data, hierarchy, encoder, device, model_path, cache_dir, cache_key)
    from src.model import get_graph_embedding as gge
    embeds, ids = [], []
    for coarse_id, graph in enumerate(hierarchy["coarse_graphs"]):
        if graph is None:
            continue
        g = graph
        if getattr(g, "x", None) is None:
            gids = g.global_id
            g = g.clone(); g.x = data.x[gids]
        emb = gge(g, encoder, device).detach().cpu().float().view(-1)
        embeds.append(emb); ids.append(int(coarse_id))
    P = torch.stack(embeds, dim=0)
    P = torch.nn.functional.normalize(P, dim=1)
    return P, ids


def fps_seeds(edge_index, num_nodes, n_seeds):
    """Farthest-point-sample seeds by BFS graph distance (spreads seeds over the query)."""
    if num_nodes <= n_seeds:
        return list(range(num_nodes))
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    ei = edge_index.cpu().numpy()
    A = csr_matrix((np.ones(ei.shape[1]*2),
                    (np.concatenate([ei[0], ei[1]]), np.concatenate([ei[1], ei[0]]))),
                   shape=(num_nodes, num_nodes))
    seeds = [0]
    dmin = shortest_path(A, indices=0, directed=False)
    dmin[~np.isfinite(dmin)] = 1e6
    for _ in range(n_seeds - 1):
        nxt = int(np.argmax(dmin))
        seeds.append(nxt)
        d = shortest_path(A, indices=nxt, directed=False); d[~np.isfinite(d)] = 1e6
        dmin = np.minimum(dmin, d)
    return seeds


def decompose(query, n_seeds, radius):
    """Split query into r-hop balls around FPS seeds -> list of sub-Data (relabeled)."""
    n = query.num_nodes
    ei = query.edge_index
    if n <= 3 or ei.numel() == 0:
        return [query]
    et = getattr(query, "edge_type", None)
    pieces = []
    for s in fps_seeds(ei, n, n_seeds):
        subset, sub_ei, _, emask = k_hop_subgraph(int(s), radius, ei, relabel_nodes=True, num_nodes=n)
        d = Data(x=query.x[subset], edge_index=sub_ei)
        if et is not None:
            d.edge_type = et[emask]
        pieces.append(d)
    return pieces


def emb_set(pieces, encoder, device):
    vs = [get_graph_embedding(p, encoder, device).detach().cpu().float().view(-1) for p in pieces]
    V = torch.stack(vs, dim=0)
    return torch.nn.functional.normalize(V, dim=1)


def ranking_from_scores(scores, coarse_ids):
    order = torch.argsort(scores, descending=True).tolist()
    return [coarse_ids[i] for i in order]


def max_true_rank(ranking, true_coarse):
    pos = {c: r for r, c in enumerate(ranking)}
    return max((pos.get(int(t), 10**9) for t in true_coarse), default=10**9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mag")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--hierarchy-path", required=True)
    ap.add_argument("--model", required=True, help="encoder checkpoint path")
    ap.add_argument("--cache-dir", default="cache/probe")
    ap.add_argument("--seeds", default="20260607,20260608")
    ap.add_argument("--sizes", default="20,50,100")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--query-types", default="single,multi_fine,multi_coarse,degree_k_hop,k_hop,random_walk")
    ap.add_argument("--budgets", default="20,100,1000")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device} loading {args.dataset}", flush=True)
    data = C.load_named_data(args.dataset, args.data_root)
    cache_key = C.safe_cache_key(args.dataset, C.path_fingerprint(args.hierarchy_path),
                                 data.num_nodes, data.edge_index.size(1))
    hierarchy = C.load_or_prepare_hierarchy(data, args.hierarchy_path, args.cache_dir, cache_key, args.dataset)
    encoder, _ = bench.load_model(args.model, data.x.size(1), device)
    encoder.eval()

    print("[probe] building partition embedding matrix", flush=True)
    P, coarse_ids = partition_embed_matrix(data, hierarchy, encoder, device, args.model, args.cache_dir, args.dataset)
    Pt = P.to(device)
    budgets = [int(b) for b in args.budgets.split(",")]

    variants = ["single", "subq8_max", "subq8_top2", "subq4_max"]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fh = open(args.output, "w", newline="", encoding="utf-8")
    cols = ["seed", "query_id", "query_type", "target_query_size", "true_coarse_count"]
    for v in variants:
        cols.append(f"max_true_rank_{v}")
        for b in budgets:
            cols.append(f"fullcov{b}_{v}")
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()

    for seed in [int(s) for s in args.seeds.split(",")]:
        for size in [int(s) for s in args.sizes.split(",")]:
            items = C.generate_cascade_queries(data, hierarchy, args.queries, size, seed, args.query_types)
            for it in items:
                if it["is_negative"]:
                    continue
                q = it["query"]; tc = it["true_coarse"]
                row = {"seed": seed, "query_id": it["query_id"], "query_type": it["query_type"],
                       "target_query_size": it["target_query_size"], "true_coarse_count": len(tc)}
                # single-vector
                zq = torch.nn.functional.normalize(get_graph_embedding(q, encoder, device).detach().cpu().float().view(1, -1), dim=1).to(device)
                sc_single = (zq @ Pt.t()).view(-1)
                # multi-vector sets
                sets = {}
                for tag, ns, rad in [("subq8", 8, 2), ("subq4", 4, 2)]:
                    V = emb_set(decompose(q, ns, rad), encoder, device).to(device)  # k x d
                    sims = V @ Pt.t()  # k x num_parts
                    sets[tag] = sims
                def record(tag, scores):
                    rk = ranking_from_scores(scores.detach().cpu(), coarse_ids)
                    mr = max_true_rank(rk, tc)
                    row[f"max_true_rank_{tag}"] = mr
                    for b in budgets:
                        row[f"fullcov{b}_{tag}"] = int(mr < b)
                record("single", sc_single)
                record("subq8_max", sets["subq8"].max(dim=0).values)
                topk = min(2, sets["subq8"].size(0))
                record("subq8_top2", sets["subq8"].topk(topk, dim=0).values.mean(dim=0))
                record("subq4_max", sets["subq4"].max(dim=0).values)
                w.writerow(row)
            fh.flush()
            print(f"[probe] seed={seed} size={size} done", flush=True)
    fh.close()
    print(f"[probe] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
