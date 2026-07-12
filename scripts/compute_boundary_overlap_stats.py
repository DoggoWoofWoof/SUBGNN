"""Correct per-partition one-hop overlap statistics.

The paper's existing `partition_overlap_stats.csv` reports
`overlap_expanded_nodes` = the union of ALL nodes in every partition a coarse
part borders (948 neighbor parts x ~969 nodes on MAG => ~920K). That is a loose
reachability bound, NOT what the solver sees. The actual cascade builds
`coarse_overlap_node_sets[p]` = only the boundary nodes that share an edge with
p (median ~4.2K on MAG). This script computes the correct, defensible numbers
directly from the cached overlap indexes for each dataset:

  partition_nodes        : nodes in the coarse part itself
  boundary_overlap_nodes : one-hop boundary nodes added by the actual operator
  expanded_part_nodes    : part + boundary overlap (what one selected part contributes)
  neighbor_parts_touched : number of distinct neighbor partitions p borders
  full_neighbor_expansion : the loose union-of-whole-neighbor-partitions bound (for contrast)

Writes runs/diagnostics/boundary_overlap_stats.csv.
"""

import argparse
import csv
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch


DEFAULT_INDEXES = {
    "cora": "runs/lightning_connected_reruns/jigsaw-cora-mf-mc-connected-gcp-cpux8-v3/overlap_cascade/*_overlap_index.pt",
    "arxiv": "runs/lcr_arxiv_v3/overlap_cascade/*_overlap_index.pt",
    "mag": "runs/lcr_mag_v4_probe/overlap_cascade/*_overlap_index.pt",
}


def pct(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def stats_block(prefix, values):
    return {
        f"{prefix}_min": min(values) if values else 0,
        f"{prefix}_median": round(st.median(values), 1) if values else 0,
        f"{prefix}_mean": round(st.mean(values), 1) if values else 0,
        f"{prefix}_p90": round(pct(values, 0.90), 1),
        f"{prefix}_max": max(values) if values else 0,
    }


def analyze(dataset, pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"[skip] {dataset}: no overlap index at {pattern}")
        return None
    path = matches[0]
    print(f"[{dataset}] loading {path}", flush=True)
    idx = torch.load(path, map_location="cpu", weights_only=False)
    part_sets = idx["coarse_part_node_sets"]
    overlap_sets = idx.get("coarse_overlap_node_sets", {})

    # node -> home partition, and partition sizes
    home = {}
    part_size = {}
    for p, nodes in part_sets.items():
        p = int(p)
        nodes = set(int(n) for n in nodes)
        part_size[p] = len(nodes)
        for n in nodes:
            home[n] = p

    part_nodes, boundary, expanded_part, neigh_parts, full_neighbor = [], [], [], [], []
    for p in sorted(part_sets):
        p = int(p)
        psize = part_size.get(p, 0)
        onodes = set(int(n) for n in overlap_sets.get(p, ()))
        # distinct neighbor partitions contributing boundary nodes + full-expansion bound
        neighbors = set()
        for v in onodes:
            h = home.get(v)
            if h is not None and h != p:
                neighbors.add(h)
        full_exp = psize + sum(part_size.get(q, 0) for q in neighbors)
        part_nodes.append(psize)
        boundary.append(len(onodes))
        expanded_part.append(psize + len(onodes))
        neigh_parts.append(len(neighbors))
        full_neighbor.append(full_exp)

    row = {"dataset": dataset, "coarse_parts": len(part_sets), "source": path}
    row.update(stats_block("partition_nodes", part_nodes))
    row.update(stats_block("boundary_overlap_nodes", boundary))
    row.update(stats_block("expanded_part_nodes", expanded_part))
    row.update(stats_block("neighbor_parts_touched", neigh_parts))
    row.update(stats_block("full_neighbor_expansion_nodes", full_neighbor))
    row["boundary_overlap_multiple"] = round(st.median(expanded_part) / max(st.median(part_nodes), 1), 2)
    print(
        f"[{dataset}] part median {row['partition_nodes_median']:.0f}, "
        f"boundary overlap median {row['boundary_overlap_nodes_median']:.0f}, "
        f"expanded part median {row['expanded_part_nodes_median']:.0f} "
        f"({row['boundary_overlap_multiple']}x); "
        f"loose full-neighbor bound median {row['full_neighbor_expansion_nodes_median']:,.0f}",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/diagnostics/boundary_overlap_stats.csv")
    parser.add_argument("--datasets", default="cora,arxiv,mag")
    args = parser.parse_args()

    rows = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        pattern = DEFAULT_INDEXES.get(ds)
        if not pattern:
            print(f"[skip] unknown dataset {ds}")
            continue
        row = analyze(ds, pattern)
        if row:
            rows.append(row)

    if rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
