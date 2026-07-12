"""Cheap inference-only probes for the coverage failure on spatially-extended queries.

Runs several mechanisms against the single-vector coarse baseline on the *same*
MAG queries as scripts/probe_multivector_ranking.py, so per-family FullCov is
directly comparable. All methods are pure encode + similarity + (sparse) matmul /
argsort. No Glasgow, no candidate assembly. Writes a per-query CSV.

Methods
  single        : current retriever — single query vector vs coarse partition embeddings.
  fine_parent   : rank the ~10k FINE partitions (median ~194 nodes) by the query vector,
                  score each coarse partition by its best fine child (scatter-max).
                  Tests finer PARTITION granularity, holding the query side fixed.
  fine_overlap  : fine_parent PLUS one-hop structure propagation over the FINE partition
                  graph (fs + beta * Ahat_fine @ fs) before mapping to coarse. This is the
                  "fine overlap" idea: expand along fine edges, which are sparse enough to
                  be selective (unlike coarse edges, ~948 neighbors each), tracing the
                  query's connected footprint.
  fine_overlap2 : two-hop fine propagation.
  subq_fine     : ColBERT-style approximation — decompose the query into subgraphs, embed
                  each, scatter-max each subvector over fine children, then MaxSim across
                  subqueries. Attacks BOTH dilution sources (query blur AND partition blur).
  diff1         : one-hop propagation over the COARSE partition graph (contrast: floods).
  stitch        : deployed ranked_neighbor_stitch over the coarse graph (budget-capped).
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_glasgow as bench
import benchmark_retrieval as bret
import benchmark_overlap_glasgow_cascade as C
from src.model import get_graph_embedding
from retrieval_strategies import ranked_neighbor_stitch
from probe_multivector_ranking import partition_embed_matrix, max_true_rank, decompose, emb_set


def build_adjacency(graph, id_list):
    """Row-normalized weighted adjacency over id_list (scipy CSR). Edges to ids
    outside id_list are ignored. Returns (Ahat, avg_degree)."""
    import scipy.sparse as sp
    id_to_row = {int(c): i for i, c in enumerate(id_list)}
    n = len(id_list)
    rows, cols, vals = [], [], []
    for u, v, d in graph.edges(data=True):
        u = int(u); v = int(v)
        ru = id_to_row.get(u); rv = id_to_row.get(v)
        if ru is not None and rv is not None and ru != rv:
            w = float(d.get("weight", 1.0))
            rows += [ru, rv]; cols += [rv, ru]; vals += [w, w]
    A = sp.csr_matrix((np.asarray(vals, dtype=np.float64),
                       (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))), shape=(n, n))
    deg = np.asarray((A > 0).sum(axis=1)).flatten()
    rowsum = np.asarray(A.sum(axis=1)).flatten(); rowsum[rowsum == 0] = 1.0
    Ahat = sp.diags(1.0 / rowsum) @ A
    return Ahat.astype(np.float32), float(deg.mean()) if n else 0.0


def rank_from_score_vec(score, id_list):
    order = np.argsort(-score, kind="stable").tolist()
    return [id_list[i] for i in order]


def fullcov_row(row, tag, ranking, true_coarse, budgets):
    mr = max_true_rank(ranking, true_coarse)
    row[f"max_true_rank_{tag}"] = mr
    for b in budgets:
        row[f"fullcov{b}_{tag}"] = int(mr < b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mag")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--hierarchy-path", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-dir", default="cache/probe")
    ap.add_argument("--seeds", default="20260607,20260608")
    ap.add_argument("--sizes", default="20,50,100")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--query-types", default="single,multi_fine,multi_coarse,degree_k_hop,k_hop,random_walk")
    ap.add_argument("--budgets", default="20,100,1000")
    ap.add_argument("--diff-beta", type=float, default=0.5)
    ap.add_argument("--subq-seeds", type=int, default=8)
    ap.add_argument("--subq-radius", type=int, default=2)
    ap.add_argument("--stitch-budget", type=int, default=200)
    ap.add_argument("--stitch-pool", type=int, default=400)
    ap.add_argument("--stitch-seeds", type=int, default=20)
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
    budgets = [int(b) for b in args.budgets.split(",")]
    beta = args.diff_beta

    print("[probe] building coarse partition embedding matrix", flush=True)
    P, coarse_ids = partition_embed_matrix(data, hierarchy, encoder, device, args.model, args.cache_dir, args.dataset)
    Pt = P.to(device)
    id_to_row = {int(c): i for i, c in enumerate(coarse_ids)}
    num = len(coarse_ids)

    print("[probe] building global fine embedding matrix", flush=True)
    fine_ids, fine_emb = bret._build_global_fine_embeddings(hierarchy, data, encoder, device)
    fine_emb = torch.nn.functional.normalize(fine_emb.float(), dim=1)
    fine_to_coarse = {int(k): int(v) for k, v in hierarchy["fine_to_coarse_map"].items()}
    keep = [i for i, f in enumerate(fine_ids) if id_to_row.get(fine_to_coarse.get(int(f), -1)) is not None]
    fine_emb_k = fine_emb[keep]
    fine_ids_k = [int(fine_ids[i]) for i in keep]
    child_row_k = torch.tensor([id_to_row[fine_to_coarse[f]] for f in fine_ids_k], dtype=torch.long)
    print(f"[probe] {num} coarse, {len(fine_ids_k)} fine (kept)", flush=True)

    print("[probe] building coarse + fine adjacency", flush=True)
    Ahat_c, deg_c = build_adjacency(hierarchy["coarse_part_graph"], coarse_ids)
    Ahat_f, deg_f = build_adjacency(hierarchy["fine_part_graph"], fine_ids_k)
    print(f"[probe] avg coarse degree={deg_c:.1f} (of {num}), avg fine degree={deg_f:.1f} (of {len(fine_ids_k)})", flush=True)
    cpg = hierarchy["coarse_part_graph"]

    def smax(fine_scores_np):
        t = torch.from_numpy(np.ascontiguousarray(fine_scores_np)).float()
        return torch.full((num,), -1e9).scatter_reduce(0, child_row_k, t, reduce="amax", include_self=True).numpy()

    variants = ["single", "fine_parent", "fine_overlap", "fine_overlap2", "subq_fine", "diff1", "stitch"]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fh = open(args.output, "w", newline="", encoding="utf-8")
    cols = ["seed", "query_id", "query_type", "target_query_size", "true_coarse_count"]
    for v in variants:
        cols.append(f"max_true_rank_{v}")
        for b in budgets:
            cols.append(f"fullcov{b}_{v}")
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()

    stitch_time = 0.0
    for seed in [int(s) for s in args.seeds.split(",")]:
        for size in [int(s) for s in args.sizes.split(",")]:
            items = C.generate_cascade_queries(data, hierarchy, args.queries, size, seed, args.query_types)
            for it in items:
                if it["is_negative"]:
                    continue
                q = it["query"]; tc = [int(t) for t in it["true_coarse"]]
                row = {"seed": seed, "query_id": it["query_id"], "query_type": it["query_type"],
                       "target_query_size": it["target_query_size"], "true_coarse_count": len(tc)}
                zq = torch.nn.functional.normalize(
                    get_graph_embedding(q, encoder, device).detach().cpu().float().view(1, -1), dim=1)
                # coarse single-vector baseline
                s = (zq.to(device) @ Pt.t()).view(-1).cpu().numpy().astype(np.float32)
                fullcov_row(row, "single", rank_from_score_vec(s, coarse_ids), tc, budgets)
                # fine similarities
                fs = (fine_emb_k @ zq.t()).view(-1).numpy().astype(np.float32)
                fullcov_row(row, "fine_parent", rank_from_score_vec(smax(fs), coarse_ids), tc, budgets)
                # fine overlap (structure propagation on the FINE graph, then max to coarse)
                hf1 = Ahat_f @ fs
                fd1 = fs + beta * hf1
                fd2 = fd1 + beta * beta * (Ahat_f @ hf1)
                fullcov_row(row, "fine_overlap", rank_from_score_vec(smax(fd1), coarse_ids), tc, budgets)
                fullcov_row(row, "fine_overlap2", rank_from_score_vec(smax(fd2), coarse_ids), tc, budgets)
                # subquery x fine-child MaxSim (ColBERT approx)
                V = emb_set(decompose(q, args.subq_seeds, args.subq_radius), encoder, device)  # [k,d]
                Vf = (V @ fine_emb_k.t()).numpy().astype(np.float32)  # [k, Nfine]
                sub_coarse = np.stack([smax(Vf[r]) for r in range(Vf.shape[0])], axis=0)  # [k, num]
                fullcov_row(row, "subq_fine", rank_from_score_vec(sub_coarse.max(axis=0), coarse_ids), tc, budgets)
                # coarse diffusion contrast
                fullcov_row(row, "diff1", rank_from_score_vec(s + beta * (Ahat_c @ s), coarse_ids), tc, budgets)
                # coarse neighbor-stitch (deployable, budget-capped)
                t0 = time.perf_counter()
                stitched = ranked_neighbor_stitch(
                    rank_from_score_vec(s, coarse_ids), args.stitch_budget, cpg,
                    seed_count=args.stitch_seeds, pool_k=args.stitch_pool)
                stitch_time += time.perf_counter() - t0
                fullcov_row(row, "stitch", stitched, tc, budgets)
                w.writerow(row)
            fh.flush()
            print(f"[probe] seed={seed} size={size} done (stitch cum {stitch_time:.1f}s)", flush=True)
    fh.close()
    print(f"[probe] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
