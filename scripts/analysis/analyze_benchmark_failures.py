"""
Summarize Jigsaw Glasgow benchmark CSVs.

This separates retrieval coverage from exact solver success, which is the main
diagnostic needed for the Arxiv k-hop results.

Usage:
    python scripts/analyze_benchmark_failures.py runs/logs/glasgow_benchmark_arxiv_all_k100_coverage_v1_epoch20probe_k100.csv
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict


def as_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def true_part_count(row):
    explicit = as_int(row.get("true_coarse_count"), default=-1)
    if explicit >= 0:
        return explicit
    return len(re.findall(r"\d+", row.get("true_coarse_indices", "")))


def mean(values):
    return statistics.fmean(values) if values else 0.0


def median(values):
    return statistics.median(values) if values else 0.0


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row.get("query_type", "unknown"), row.get("solver_mode", "unknown"))
        groups[key].append(row)

    print("query_type,solver_mode,n,solved,fullcov_at_k,fullcov_at_k_pct,candidate_fullcov,candidate_fullcov_pct,candidate_fine_fullcov,candidate_fine_fullcov_pct,solver_failed_after_candidate_fullcov,solver_failed_after_fullcov,avg_recall_at_k,avg_true_parts,avg_true_fine_parts,avg_stitched_nodes,avg_pre_prune_nodes,avg_pruned_nodes,median_solver_s")
    for key in sorted(groups):
        qtype, mode = key
        group = groups[key]
        solved = [row for row in group if as_bool(row.get("perfect_solution_found"))]
        fullcov = [
            row for row in group
            if as_bool(row.get("fullcov_at_k"))
            or as_bool(row.get("retrieval_complete_at_k"))
            or as_float(row.get("coarse_recall_at_k")) >= 1.0
        ]
        solver_failed_after_fullcov = [
            row for row in fullcov if not as_bool(row.get("perfect_solution_found"))
        ]
        candidate_fullcov = [
            row for row in group
            if as_bool(row.get("candidate_fullcov"))
        ]
        candidate_fine_fullcov = [
            row for row in group
            if as_bool(row.get("candidate_fine_fullcov"))
        ]
        solver_failed_after_candidate_fullcov = [
            row for row in candidate_fullcov if not as_bool(row.get("perfect_solution_found"))
        ]
        recalls = [as_float(row.get("coarse_recall_at_k")) for row in group]
        true_parts = [true_part_count(row) for row in group]
        true_fine_parts = [as_int(row.get("true_fine_count")) for row in group]
        stitched_nodes = [as_int(row.get("stitched_nodes")) for row in group if as_int(row.get("stitched_nodes")) > 0]
        pre_prune_nodes = [as_int(row.get("pre_prune_stitched_nodes")) for row in group if as_int(row.get("pre_prune_stitched_nodes")) > 0]
        pruned_nodes = [as_int(row.get("pruned_stitched_nodes")) for row in group if as_int(row.get("pruned_stitched_nodes")) > 0]
        solver_times = [as_float(row.get("solver_time")) for row in group]

        print(
            f"{qtype},{mode},{len(group)},{len(solved)},{len(fullcov)},"
            f"{(100.0 * len(fullcov) / len(group)) if group else 0.0:.1f},"
            f"{len(candidate_fullcov)},"
            f"{(100.0 * len(candidate_fullcov) / len(group)) if group else 0.0:.1f},"
            f"{len(candidate_fine_fullcov)},"
            f"{(100.0 * len(candidate_fine_fullcov) / len(group)) if group else 0.0:.1f},"
            f"{len(solver_failed_after_candidate_fullcov)},"
            f"{len(solver_failed_after_fullcov)},"
            f"{mean(recalls):.4f},{mean(true_parts):.2f},{mean(true_fine_parts):.2f},"
            f"{mean(stitched_nodes):.1f},{mean(pre_prune_nodes):.1f},{mean(pruned_nodes):.1f},"
            f"{median(solver_times):.2f}"
        )


def print_failures(rows):
    failures = [row for row in rows if not as_bool(row.get("perfect_solution_found"))]
    if not failures:
        return

    print("\nfailed_rows")
    print("query_name,query_type,query_nodes,true_parts,true_fine_parts,fullcov_at_k,candidate_fullcov,candidate_fine_fullcov,recall_at_k,missed_at_k,candidate_missed,candidate_fine_missed,solver_level,stitched_nodes,pre_prune_nodes,pruned_nodes,solver_s")
    for row in failures:
        fullcov = (
            as_bool(row.get("fullcov_at_k"))
            or as_bool(row.get("retrieval_complete_at_k"))
            or as_float(row.get("coarse_recall_at_k")) >= 1.0
        )
        print(
            f"{row.get('query_name','')},{row.get('query_type','')},"
            f"{row.get('query_nodes','')},{true_part_count(row)},{as_int(row.get('true_fine_count'))},"
            f"{fullcov},"
            f"{as_bool(row.get('candidate_fullcov'))},"
            f"{as_bool(row.get('candidate_fine_fullcov'))},"
            f"{row.get('coarse_recall_at_k','')},"
            f"\"{row.get('missed_coarse_at_k','')}\","
            f"\"{row.get('candidate_missed_coarse','')}\","
            f"\"{row.get('candidate_missed_fine','')}\","
            f"{row.get('solver_level','')},{row.get('stitched_nodes','')},"
            f"{row.get('pre_prune_stitched_nodes','')},{row.get('pruned_stitched_nodes','')},"
            f"{as_float(row.get('solver_time')):.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark retrieval/solver failures.")
    parser.add_argument("csv_path", help="Benchmark CSV path")
    args = parser.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    summarize(rows)
    print_failures(rows)


if __name__ == "__main__":
    main()
