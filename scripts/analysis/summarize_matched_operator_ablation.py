"""Summarize operator ablations at a matched fraction of coarse partitions.

The raw cascade CSVs stop evaluating a query after its first successful budget.
For a requested reporting budget, this script therefore selects the first solved
row at or below that budget, or the largest attempted row at or below the budget
when the query remains unsolved. Only planted positive queries contribute to the
solve-rate and candidate-size statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VARIANTS = ("full", "no_overlap", "no_signature", "no_components", "no_exact_label")
DEFAULT_DATASETS = {
    "arxiv": {
        "results_dir": Path("runs/arxiv_design_ablation_v1_dl/results"),
        "total_partitions": 200,
    },
    "mag": {
        "results_dir": Path("runs/mag_design_ablation_v2_dl/results"),
        "total_partitions": 2000,
    },
}
QUERY_KEY = ["query_type", "target_query_size", "query_id"]


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def summarize_file(path: Path, reporting_budget: int) -> dict[str, float | int | str]:
    frame = pd.read_csv(path)
    positives = frame.loc[~_as_bool(frame["is_negative"])].copy()
    positives = positives.loc[positives["budget"] <= reporting_budget]
    if positives.empty:
        raise ValueError(f"No positive rows at or below K={reporting_budget}: {path}")

    selected_rows = []
    for _, group in positives.groupby(QUERY_KEY, sort=False, dropna=False):
        group = group.sort_values("budget")
        solved = group.loc[_as_bool(group["solver_found"])]
        selected_rows.append(solved.iloc[0] if not solved.empty else group.iloc[-1])

    selected = pd.DataFrame(selected_rows)
    solved = _as_bool(selected["solver_found"])
    timed_out = _as_bool(selected["solver_timed_out"])
    return {
        "positive_queries": len(selected),
        "solved": int(solved.sum()),
        "solve_rate_percent": 100.0 * float(solved.mean()),
        "timed_out": int(timed_out.sum()),
        "mean_pruned_candidate_nodes": float(selected["pruned_candidate_nodes"].mean()),
        "median_pruned_candidate_nodes": float(selected["pruned_candidate_nodes"].median()),
        "candidate_semantics": "first solved row at or below K; otherwise largest attempted row at or below K",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/paper_results/ablations/operator_ablation_half_budget_summary.csv"),
    )
    args = parser.parse_args()

    rows = []
    for dataset, config in DEFAULT_DATASETS.items():
        total_partitions = int(config["total_partitions"])
        reporting_budget = total_partitions // 2
        for variant in VARIANTS:
            source = Path(config["results_dir"]) / f"{dataset}_ablation_{variant}_per_query.csv"
            row = summarize_file(source, reporting_budget)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "total_coarse_partitions": total_partitions,
                    "reporting_budget": reporting_budget,
                    "budget_fraction": reporting_budget / total_partitions,
                    **row,
                    "source_file": source.as_posix(),
                }
            )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
