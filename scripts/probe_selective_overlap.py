"""Offline probe: recall/size tradeoff of selective overlap.

Runs entirely from cached assets (overlap index, label tokens, query set) with no
encoder, no raw graph, and no exact solver -- because the metrics that matter for
the overlap operator (does the true match survive into the candidate? how big is
the candidate?) depend only on partition/overlap membership and node labels.

Two regimes, because overlap plays a different role in each:

  complete   -- seed = all true partitions + random noise partitions up to a realistic
                budget. Retrieval found the answer's partitions (the common case), so
                recall is guaranteed and this isolates the SIZE question: does selective
                overlap shrink the candidate for free? (selective overlap only trims
                overlap nodes, never partition-interior nodes, so query nodes always
                survive -> recall is provably preserved here.)

  incomplete -- seed = true partitions with a fraction dropped. Tests whether overlap
                can REPAIR missing partitions. Key finding: one-hop overlap recovers
                only boundary nodes, not the interior of a fully-unselected partition,
                so even blunt overlap mostly fails here -- meaning selective overlap
                sacrifices no recall that blunt overlap was actually providing.

Reported per (query_type, policy):
  pruned_fullcov  : fraction of queries whose full node set survives overlap + label pruning
  overlap_nodes   : candidate size BEFORE standard label pruning (the candidate-build / latency driver)
  pruned_nodes    : candidate size AFTER standard label pruning (what the solver faces)

Runs entirely from cached assets and calls the SAME selective_overlap_for_parts used
by the production cascade, so it validates the real code path.
"""

import argparse
import csv
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_overlap_glasgow_cascade import (  # noqa: E402
    build_overlap_neighbor_index,
    candidate_nodes_for_parts,
    prune_nodes_by_query_label_tokens,
    selective_overlap_for_parts,
)


POLICIES = {
    "no_overlap": {"use_overlap": False},
    "blunt": {},
    "topk8": {"max_parts": 8},
    "topk8_label": {"max_parts": 8, "label_compatible": True},
    # recall-expansion (random-walk fix): add full nodes of the top-N partitions
    # most strongly bridged (by boundary support) to the selected set -- i.e. the
    # mid-path partitions a diffuse walk missed. Support-ranked + bounded so it
    # cannot explode on MAG's near-complete coarse graph.
    "bridge8": {"bridge_infill_top": 8},
    "bridge16": {"bridge_infill_top": 16},
    "bridge32": {"bridge_infill_top": 32},
    "bridge16_only": {"boundary_overlap": False, "bridge_infill_top": 16},
}


def median(values):
    return statistics.median(values) if values else 0.0


def fullcov(query_nodes, candidate):
    return set(query_nodes).issubset(candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overlap-index", default="runs/lcr_mag_v4_probe/overlap_cascade/49c19662750e1cef_overlap_index.pt")
    parser.add_argument("--label-tokens", default="runs/lcr_mag_v4_probe/overlap_cascade/49c19662750e1cef_feature_label_tokens.pt")
    parser.add_argument("--queries", default="runs/quarantine_disconnected_query_caches/runs__query_cache__mag_all_q50_sizes20_50_100__mag_all_seed20260607_queries.pt")
    parser.add_argument("--out", default="runs/diagnostics/selective_overlap_probe.csv")
    parser.add_argument("--mode", choices=["complete", "incomplete"], default="complete",
                        help="complete: true parts + noise up to --budget-parts (isolates size at fixed recall); "
                             "incomplete: drop a fraction of true parts (tests overlap recall repair)")
    parser.add_argument("--budget-parts", type=int, default=100, help="complete mode: total selected partitions (true + noise) to mimic a realistic retrieval budget")
    parser.add_argument("--keep-frac", type=float, default=0.5, help="incomplete mode: fraction of each query's true coarse parts kept as the seed")
    parser.add_argument("--num-parts", type=int, default=2000)
    parser.add_argument("--max-queries", type=int, default=0, help="cap number of queries evaluated (0=all); use for fast timing checks")
    parser.add_argument("--family", default="", help="restrict to one query family (e.g. random_walk); empty = all multi-part families")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--positives-only", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Loading overlap index: {args.overlap_index}", flush=True)
    t = time.perf_counter()
    idx = torch.load(args.overlap_index, map_location="cpu", weights_only=False)
    hierarchy = {
        "coarse_part_node_sets": {int(k): set(int(x) for x in v) for k, v in idx["coarse_part_node_sets"].items()},
        "coarse_overlap_node_sets": idx["coarse_overlap_node_sets"],
    }
    print(f"  loaded in {time.perf_counter()-t:.1f}s; {len(hierarchy['coarse_part_node_sets'])} parts", flush=True)

    build_overlap_neighbor_index(hierarchy)

    print(f"Loading label tokens: {args.label_tokens}", flush=True)
    label_tokens = torch.load(args.label_tokens, map_location="cpu", weights_only=False).long()

    print(f"Loading queries: {args.queries}", flush=True)
    queries = torch.load(args.queries, map_location="cpu", weights_only=False)

    # Alignment guard: the query set MUST be generated against the same hierarchy as
    # the overlap index, or recall/recovery/fullcov metrics are meaningless (node IDs
    # land in unrelated partitions). Build time and overlap size remain valid either way.
    part_sets_check = hierarchy["coarse_part_node_sets"]
    sample = [q for q in queries if q.get("expected_match", True)][:50]
    cov = []
    for q in sample:
        qn = [int(n) for n in q["query_nodes"].tolist()]
        tc = set(int(x) for x in q["true_coarse"])
        seed = set()
        for p in tc:
            seed |= part_sets_check.get(int(p), set())
        if qn:
            cov.append(sum(1 for n in qn if n in seed) / len(qn))
    align = sum(cov) / len(cov) if cov else 0.0
    if align < 0.9:
        print(
            f"\n*** WARNING: query/index hierarchy MISMATCH (mean true-part coverage "
            f"{align:.1%}). RECALL/recovery/fullcov metrics are INVALID; only build-time "
            f"and overlap-size columns are trustworthy. Regenerate queries against this "
            f"index's hierarchy for valid recall numbers. ***\n",
            flush=True,
        )
    else:
        print(f"Query/index alignment OK (true-part coverage {align:.1%}).", flush=True)

    rng = random.Random(args.seed)
    part_sets = hierarchy["coarse_part_node_sets"]
    # accumulator[(query_type, policy)] -> dict of lists
    acc = defaultdict(lambda: {"fullcov": [], "overlap": [], "pruned": [], "dropped": [], "recover": [], "build_ms": []})

    used = 0
    for q in queries:
        if args.max_queries and used >= args.max_queries:
            break
        if args.positives_only and not q.get("expected_match", True):
            continue
        if args.family and q.get("query_type", "") != args.family:
            continue
        true_coarse = sorted(int(x) for x in q["true_coarse"])
        if len(true_coarse) < 2:
            continue
        query_nodes = [int(n) for n in q["query_nodes"].tolist()]
        query = q["query"]
        qtype = q.get("query_type", "?")

        if args.mode == "complete":
            # Retrieval found the answer's partitions; pad with noise to a budget.
            noise_pool = [p for p in range(args.num_parts) if p not in set(true_coarse)]
            rng.shuffle(noise_pool)
            need = max(0, args.budget_parts - len(true_coarse))
            seed_parts = sorted(set(true_coarse) | set(noise_pool[:need]))
            dropped = 0
        else:
            keep_n = max(1, round(args.keep_frac * len(true_coarse)))
            shuffled = list(true_coarse)
            rng.shuffle(shuffled)
            seed_parts = sorted(shuffled[:keep_n])
            dropped = len(true_coarse) - keep_n
        used += 1

        # query nodes lost to the dropped partitions (what overlap must recover)
        seed_nodes = set()
        for p in seed_parts:
            seed_nodes |= part_sets.get(int(p), set())
        lost_nodes = [n for n in query_nodes if n not in seed_nodes]

        for name, policy in POLICIES.items():
            build_start = time.perf_counter()
            cand = selective_overlap_for_parts(
                seed_parts, hierarchy, policy=policy, query=query, label_tokens=label_tokens
            )
            pruned = prune_nodes_by_query_label_tokens(set(cand), query, label_tokens)
            build_ms = (time.perf_counter() - build_start) * 1000.0
            overlap_size = len(cand)
            rec = acc[(qtype, name)]
            rec["build_ms"].append(build_ms)
            rec["fullcov"].append(1 if fullcov(query_nodes, pruned) else 0)
            rec["overlap"].append(overlap_size)
            rec["pruned"].append(len(pruned))
            rec["dropped"].append(dropped)
            if lost_nodes:
                recovered = sum(1 for n in lost_nodes if n in pruned)
                rec["recover"].append(recovered / len(lost_nodes))

    detail = (f"budget_parts={args.budget_parts}" if args.mode == "complete"
              else f"keep_frac={args.keep_frac}")
    print(f"\nEvaluated {used} multi-part positive queries (mode={args.mode}, {detail}).\n", flush=True)

    rows = []
    for (qtype, policy), rec in sorted(acc.items()):
        n = len(rec["fullcov"])
        rows.append({
            "query_type": qtype,
            "policy": policy,
            "queries": n,
            "avg_dropped_true_parts": round(statistics.mean(rec["dropped"]), 2) if n else 0,
            "pruned_fullcov_rate": round(sum(rec["fullcov"]) / n, 4) if n else 0,
            "lost_node_recovery_frac": round(statistics.mean(rec["recover"]), 4) if rec["recover"] else 0,
            "median_candidate_build_ms": round(median(rec["build_ms"]), 2),
            "median_overlap_nodes": round(median(rec["overlap"]), 1),
            "median_pruned_nodes": round(median(rec["pruned"]), 1),
            "mean_pruned_nodes": round(statistics.mean(rec["pruned"]), 1) if n else 0,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} rows)\n")

    # Digest: aggregate across query types, ordered by policy.
    agg = defaultdict(lambda: {"fullcov": [], "overlap": [], "pruned": [], "recover": [], "build_ms": []})
    for (qtype, policy), rec in acc.items():
        for k in ("fullcov", "overlap", "pruned", "recover", "build_ms"):
            agg[policy][k].extend(rec[k])
    print("=== Selective-overlap probe digest ===")
    print(f"{'policy':16}{'fullcov':>9}{'lostNodeRecov':>14}{'med_build_ms':>13}{'med_overlap':>13}{'med_pruned':>12}")
    for policy in POLICIES:
        rec = agg.get(policy)
        if not rec or not rec["fullcov"]:
            continue
        fc = sum(rec["fullcov"]) / len(rec["fullcov"])
        rcv = (statistics.mean(rec["recover"]) if rec["recover"] else 0.0)
        print(f"{policy:16}{fc*100:8.1f}%{rcv*100:13.1f}%{median(rec['build_ms']):>13,.1f}{median(rec['overlap']):>13,.0f}{median(rec['pruned']):>12,.0f}")


if __name__ == "__main__":
    main()
