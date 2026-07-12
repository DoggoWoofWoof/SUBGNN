"""
Compare Jigsaw Glasgow benchmark CSVs at the run level.

Example:
    python scripts/compare_benchmark_csvs.py \
        --run v2_k125=runs/logs/glasgow_benchmark_arxiv_k_hop_k125_coverage_v2_allpos_fresh_fine_boundary_k125_prune_fixed_q30_seed42.csv \
        --run v2_k150=runs/logs/glasgow_benchmark_arxiv_k_hop_k150_coverage_v2_allpos_fresh_fine_boundary_k150_prune_fixed_q30_seed42.csv
"""

import argparse
import csv
import statistics
from collections import Counter


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


def mean(values):
    return statistics.fmean(values) if values else 0.0


def median(values):
    return statistics.median(values) if values else 0.0


def summarize(label, path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    stitch = [row for row in rows if row.get("solver_mode") == "stitch"]
    oracle = [row for row in rows if row.get("solver_mode") == "oracle"]
    solved = [row for row in stitch if as_bool(row.get("perfect_solution_found"))]
    oracle_solved = [row for row in oracle if as_bool(row.get("perfect_solution_found"))]
    fullcov = [row for row in stitch if as_bool(row.get("fullcov_at_k"))]
    has_score_pool = bool(stitch and "coarse_score_pool_fullcov" in stitch[0])
    has_expanded = bool(stitch and "expanded_coarse_fullcov" in stitch[0])
    has_mc = bool(stitch and "mc_dropout_seed_fullcov" in stitch[0])
    score_pool_fullcov = [
        row for row in stitch
        if as_bool(row.get("coarse_score_pool_fullcov") if has_score_pool else row.get("fullcov_at_k"))
    ]
    expanded_fullcov = [
        row for row in stitch
        if as_bool(row.get("expanded_coarse_fullcov") if has_expanded else row.get("candidate_fullcov", row.get("fullcov_at_k")))
    ]
    mc_fullcov = [
        row for row in stitch
        if as_bool(row.get("mc_dropout_seed_fullcov") if has_mc else False)
    ]
    candidate_fullcov = [row for row in stitch if as_bool(row.get("candidate_fullcov"))]
    candidate_fine_fullcov = [row for row in stitch if as_bool(row.get("candidate_fine_fullcov"))]
    solver_failed_after_candidate_fullcov = [
        row for row in candidate_fullcov if not as_bool(row.get("perfect_solution_found"))
    ]
    recalls = [as_float(row.get("coarse_recall_at_k")) for row in stitch]
    pre_prune_nodes = [
        as_int(row.get("pre_prune_stitched_nodes"))
        for row in stitch
        if as_int(row.get("pre_prune_stitched_nodes")) > 0
    ]
    pruned_nodes = [
        as_int(row.get("pruned_stitched_nodes"))
        for row in stitch
        if as_int(row.get("pruned_stitched_nodes")) > 0
    ]
    solver_times = [as_float(row.get("solver_time")) for row in stitch]
    timeouts = [row for row in stitch if as_bool(row.get("solver_timed_out"))]
    levels = Counter(row.get("solver_level", "") for row in stitch)

    return {
        "label": label,
        "path": path,
        "n": len(stitch),
        "solved": len(solved),
        "oracle_n": len(oracle),
        "oracle_solved": len(oracle_solved),
        "fullcov": len(fullcov),
        "score_pool_fullcov": len(score_pool_fullcov),
        "expanded_fullcov": len(expanded_fullcov),
        "mc_fullcov": len(mc_fullcov),
        "candidate_fullcov": len(candidate_fullcov),
        "candidate_fine_fullcov": len(candidate_fine_fullcov),
        "solver_failed_after_candidate_fullcov": len(solver_failed_after_candidate_fullcov),
        "avg_recall": mean(recalls),
        "avg_pre_prune_nodes": mean(pre_prune_nodes),
        "avg_pruned_nodes": mean(pruned_nodes),
        "median_solver_s": median(solver_times),
        "timeouts": len(timeouts),
        "levels": dict(sorted(levels.items())),
        "misses": [
            {
                "query_name": row.get("query_name"),
                "recall": row.get("coarse_recall_at_k"),
                "missed": row.get("missed_coarse_at_k"),
                "candidate_missed": row.get("candidate_missed_coarse"),
                "expanded_missed": row.get("expanded_missed_coarse"),
                "solver_level": row.get("solver_level"),
            }
            for row in stitch
            if not as_bool(row.get("perfect_solution_found"))
        ],
    }


def print_markdown(summaries):
    print("| Run | N | Stitch Solved | FullCov@SeedK | MC Seed FullCov | ScorePool FullCov | Expanded FullCov | Candidate FullCov | Fine Candidate FullCov | Oracle Solved | Avg Recall@SeedK | Avg Pre-Prune Nodes | Avg Pruned Nodes | Median Solver s | Timeouts |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in summaries:
        print(
            f"| {s['label']} | {s['n']} | {s['solved']}/{s['n']} | "
            f"{s['fullcov']}/{s['n']} | {s['mc_fullcov']}/{s['n']} | {s['score_pool_fullcov']}/{s['n']} | "
            f"{s['expanded_fullcov']}/{s['n']} | {s['candidate_fullcov']}/{s['n']} | "
            f"{s['candidate_fine_fullcov']}/{s['n']} | "
            f"{s['oracle_solved']}/{s['oracle_n']} | {s['avg_recall']:.4f} | "
            f"{s['avg_pre_prune_nodes']:.1f} | {s['avg_pruned_nodes']:.1f} | "
            f"{s['median_solver_s']:.2f} | {s['timeouts']} |"
        )

    print("\nMisses:")
    for s in summaries:
        if not s["misses"]:
            print(f"- {s['label']}: none")
            continue
        for miss in s["misses"]:
            print(
                f"- {s['label']} {miss['query_name']}: recall={miss['recall']} "
                f"missed={miss['missed']} expanded_missed={miss['expanded_missed']} "
                f"level={miss['solver_level']}"
            )


def parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be label=path")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--run must be label=path")
    return label, path


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark CSV summaries.")
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    args = parser.parse_args()

    summaries = [summarize(label, path) for label, path in args.run]
    print_markdown(summaries)


if __name__ == "__main__":
    main()
