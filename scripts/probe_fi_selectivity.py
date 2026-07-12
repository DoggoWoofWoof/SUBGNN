"""Does the label-selectivity crossover generalize beyond MAG?

Local, retrieval-only (no Glasgow). On Cora/Arxiv, compares FeatureIndex under
near-unique FEATURE labels vs realistic CLASS labels vs the learned retriever,
on the same query families. If FeatureIndex is strong under feature labels but
weak under class labels (while the learned retriever holds up), the crossover
generalizes and a label-selectivity selector is a real portfolio result.
"""
from __future__ import annotations
import argparse, csv, os, sys, statistics as st
from collections import defaultdict
import torch

# Trusted local checkpoints/hierarchies contain PyG objects; torch>=2.6 defaults
# weights_only=True and rejects them. These are our own files -> load fully.
_orig_torch_load = torch.load
def _torch_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_glasgow as bench
import benchmark_overlap_glasgow_cascade as C
from src.model import get_graph_embedding


def max_true_rank(ranking, true_coarse):
    pos = {int(c): r for r, c in enumerate(ranking)}
    return max((pos.get(int(t), 10**9) for t in true_coarse), default=10**9)


def query_global_ids(query, item):
    for obj, attr in ((query, "global_id"), (query, "n_id"), (query, "node_index")):
        v = getattr(obj, attr, None)
        if v is not None:
            return torch.as_tensor(v).long().view(-1)
    for k in ("query_nodes", "global_nodes", "node_ids", "nodes"):
        if k in item and item[k] is not None:
            return torch.as_tensor(item[k]).long().view(-1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--hierarchy-path", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--cache-dir", default="cache/fi_probe")
    ap.add_argument("--seeds", default="20260607,20260608")
    ap.add_argument("--sizes", default="20,50,100")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--query-types", default="positive")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    device = torch.device("cpu")
    data = C.load_named_data(args.dataset, args.data_root)
    hpath = args.hierarchy_path or f"{args.dataset}_hierarchies_probe.pt"
    cache_key = C.safe_cache_key(args.dataset, C.path_fingerprint(hpath), data.num_nodes, data.edge_index.size(1))
    hierarchy = C.load_or_prepare_hierarchy(data, hpath, args.cache_dir, cache_key, args.dataset)
    if "coarse_part_node_sets" not in hierarchy:
        hierarchy["coarse_part_node_sets"] = {
            int(pid): set(int(n) for n in torch.as_tensor(nodes).view(-1).tolist())
            for pid, nodes in hierarchy["coarse_part_nodes_map"].items()
        }
    ncoarse = len(hierarchy["coarse_part_node_sets"])
    print(f"[fi] {args.dataset}: {data.num_nodes} nodes, {ncoarse} coarse partitions", flush=True)

    # label selectivity statistic: distinct labels and partitions-per-label
    ydist = int(torch.unique(data.y.view(-1)).numel()) if getattr(data, "y", None) is not None else -1
    print(f"[fi] distinct class labels (data.y): {ydist}", flush=True)

    fi_feat = C.build_coarse_feature_index(data, hierarchy, args.cache_dir, cache_key, node_labels_override=None)
    fi_class = None
    if getattr(data, "y", None) is not None:
        fi_class = C.build_coarse_feature_index(data, hierarchy, args.cache_dir, cache_key + "_cls",
                                                node_labels_override=data.y.view(-1))

    encoder = None
    P = None; coarse_ids = fi_feat[2]
    if args.model:
        try:
            from probe_multivector_ranking import partition_embed_matrix
            encoder, _ = bench.load_model(args.model, data.x.size(1), device); encoder.eval()
            P, coarse_ids = partition_embed_matrix(data, hierarchy, encoder, device, args.model, args.cache_dir, args.dataset)
            print(f"[fi] neural model loaded; {P.size(0)} partition embeddings", flush=True)
        except Exception as e:
            print(f"[fi] neural skipped: {type(e).__name__}: {e}", flush=True)
            encoder = None

    methods = ["fi_feature"] + (["fi_class"] if fi_class else []) + (["neural"] if encoder is not None else [])
    budgets = sorted({max(1, ncoarse // 10), max(1, ncoarse // 4), max(1, ncoarse // 2), ncoarse})
    rows = []
    gid_ok = True
    for seed in [int(s) for s in args.seeds.split(",")]:
        for size in [int(s) for s in args.sizes.split(",")]:
            items = C.generate_cascade_queries(data, hierarchy, args.queries, size, seed, args.query_types)
            for it in items:
                if it["is_negative"]:
                    continue
                q = it["query"]; tc = [int(t) for t in it["true_coarse"]]
                row = {"seed": seed, "query_type": it["query_type"], "size": it["target_query_size"],
                       "true_coarse_count": len(tc)}
                r_feat = C.rank_by_feature_index(q, fi_feat)  # feature labels
                row["max_true_rank_fi_feature"] = max_true_rank(r_feat, tc)
                if fi_class:
                    gids = query_global_ids(q, it)
                    if gids is None:
                        gid_ok = False
                        row["max_true_rank_fi_class"] = 10**9
                    else:
                        qlab = data.y.view(-1)[gids].tolist()
                        r_cls = C.rank_by_feature_index(q, fi_class, query_labels=qlab)
                        row["max_true_rank_fi_class"] = max_true_rank(r_cls, tc)
                if encoder is not None:
                    zq = torch.nn.functional.normalize(get_graph_embedding(q, encoder, device).detach().cpu().float().view(1, -1), dim=1)
                    s = (zq @ torch.nn.functional.normalize(P.float(), dim=1).t()).view(-1)
                    order = torch.argsort(s, descending=True).tolist()
                    r_neu = [coarse_ids[i] for i in order]
                    row["max_true_rank_neural"] = max_true_rank(r_neu, tc)
                rows.append(row)
            print(f"[fi] seed={seed} size={size} done", flush=True)
    if not gid_ok:
        print("[fi] WARNING: could not resolve query global ids -> fi_class invalid", flush=True)

    # report per-family median rank + FullCov at a mid budget
    fams = ["single", "multi_fine", "multi_coarse", "degree_k_hop", "k_hop", "random_walk"]
    midb = budgets[len(budgets) // 2]
    byfam = defaultdict(list)
    for r in rows:
        byfam[r["query_type"]].append(r)
    print(f"\n=== {args.dataset}: median max_true_coarse_rank  (of {ncoarse}); [FullCov@{midb} %] ===")
    hdr = f"{'family':13s}" + "".join(f"{m:>20s}" for m in methods)
    print(hdr)
    for fam in fams:
        rs = byfam.get(fam, [])
        if not rs: continue
        cells = []
        for m in methods:
            key = f"max_true_rank_{m}"
            vals = [r[key] for r in rs if key in r]
            med = int(st.median(vals)) if vals else -1
            fc = 100.0 * sum(1 for v in vals if v < midb) / max(len(vals), 1)
            cells.append(f"{med} [{fc:.0f}]")
        print(f"{fam:13s}" + "".join(f"{c:>20s}" for c in cells))
    allr = [r for fam in fams for r in byfam.get(fam, [])]
    cells = []
    for m in methods:
        key = f"max_true_rank_{m}"; vals = [r[key] for r in allr if key in r]
        med = int(st.median(vals)) if vals else -1
        fc = 100.0 * sum(1 for v in vals if v < midb) / max(len(vals), 1)
        cells.append(f"{med} [{fc:.0f}]")
    print(f"{'OVERALL':13s}" + "".join(f"{c:>20s}" for c in cells))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        cols = ["seed", "query_type", "size", "true_coarse_count"] + [f"max_true_rank_{m}" for m in methods]
        with open(args.output, "w", newline="", encoding="utf-8") as fh:
            wtr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); wtr.writeheader(); wtr.writerows(rows)
        print(f"[fi] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
