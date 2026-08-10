"""Summarize production solve and cost at matched reporting budgets.

Each cascade CSV contains one row per attempted budget and stops after the first
solve or timeout. This reducer reports the terminal pruned candidate domain at
that stopping row, while candidate-construction and solver times are cumulative
over every attempted budget through the paper endpoint. The peak intermediate
domain is retained as a diagnostic but is not the paper's candidate column.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "hybrid": "Jigsaw",
    "coarse_mean_rrf": "Mean-RRF",
    "mean_feature": "MeanFeat",
    "topo_feature": "TopoFeat",
    "random": "Random",
    "all": "FilterAll",
}
EXPECTED_SOLVED = {
    ("Cora", "Jigsaw"): 1696,
    ("Cora", "Mean-RRF"): 1688,
    ("Cora", "MeanFeat"): 1578,
    ("Cora", "TopoFeat"): 575,
    ("Cora", "Random"): 588,
    ("Cora", "FilterAll"): 1800,
    ("Arxiv", "Jigsaw"): 1670,
    ("Arxiv", "Mean-RRF"): 1692,
    ("Arxiv", "MeanFeat"): 1615,
    ("Arxiv", "TopoFeat"): 599,
    ("Arxiv", "Random"): 706,
    ("Arxiv", "FilterAll"): 1800,
}
EXPECTED_FAMILIES = {
    "single",
    "k_hop",
    "degree_k_hop",
    "multi_fine",
    "multi_coarse",
    "random_walk",
}
EXPECTED_SIZES = {20, 50, 100}
ROOT = Path(__file__).resolve().parents[2]
QUERY_PAYLOAD_AUDIT = (
    ROOT / "benchmarks" / "paper_results" / "final_results" / "query_payload_v1_validation.json"
)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def _seed(path: Path) -> int:
    match = re.search(r"_s(20\d{6})_", path.name)
    if not match:
        raise ValueError(f"Cannot infer seed from {path}")
    return int(match.group(1))


def _load_results(results_dir: Path) -> pd.DataFrame:
    paths = sorted(
        path
        for path in results_dir.glob("*_per_query.csv")
        if "_partial_per_query" not in path.name
    )
    if len(paths) != 12:
        raise ValueError(f"Expected 12 final per-query CSVs in {results_dir}, found {len(paths)}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["seed"] = _seed(path)
        frame["source_file"] = path.as_posix()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _reduce_dataset(dataset: str, results_dir: Path, reporting_budget: int, full_budget: int):
    frame = _load_results(results_dir)
    required = {
        "query_number",
        "query_id",
        "query_type",
        "target_query_size",
        "expected_match",
        "is_negative",
        "budget",
        "method",
        "solver_found",
        "solver_timed_out",
        "pruned_candidate_nodes",
        "candidate_time_seconds",
        "solver_time_seconds",
        "retrieval_time_seconds",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset}: missing columns {missing}")
    if not _as_bool(frame["expected_match"]).all() or _as_bool(frame["is_negative"]).any():
        raise ValueError(f"{dataset}: matched-cost run must contain positive queries only")
    if "query_pruning_source" in frame:
        if set(frame["query_pruning_source"].astype(str)) != {"query_payload_v1"}:
            raise ValueError(f"{dataset}: every row must use query_payload_v1 pruning")
    else:
        audit = json.loads(QUERY_PAYLOAD_AUDIT.read_text(encoding="utf-8"))
        dataset_audit = audit["datasets"][dataset.lower()]
        if (
            audit.get("status") != "pass"
            or dataset_audit.get("positive_queries") != 1800
            or dataset_audit.get("positive_payload_equals_legacy_tokens") != 1800
            or "query_label_pruning" not in frame
            or not _as_bool(frame["query_label_pruning"]).all()
        ):
            raise ValueError(
                f"{dataset}: missing query_pruning_source without a complete "
                "positive-token equivalence audit"
            )
        frame["query_pruning_source"] = "query_payload_v1_equivalent_positive"

    frame["method_label"] = frame["method"].map(METHOD_LABELS)
    if frame["method_label"].isna().any():
        bad = sorted(frame.loc[frame["method_label"].isna(), "method"].astype(str).unique())
        raise ValueError(f"{dataset}: unknown methods {bad}")

    summary_rows = []
    detail_rows = []
    for method_label, method_rows in frame.groupby("method_label", sort=False):
        endpoint = full_budget if method_label == "FilterAll" else reporting_budget
        method_rows = method_rows.loc[method_rows["budget"] <= endpoint].copy()
        selected = []
        query_key = ["source_file", "seed", "query_number", "query_id"]
        for key, group in method_rows.groupby(query_key, sort=False, dropna=False):
            group = group.sort_values("budget")
            stop = group.loc[_as_bool(group["solver_found"]) | _as_bool(group["solver_timed_out"])]
            terminal = stop.iloc[0] if not stop.empty else group.iloc[-1]
            attempted = group.loc[group["budget"] <= int(terminal["budget"])]
            row = terminal.to_dict()
            row.update(
                {
                    "dataset": dataset,
                    "peak_pruned_candidate_nodes": float(attempted["pruned_candidate_nodes"].max()),
                    "matched_candidate_time_seconds": float(attempted["candidate_time_seconds"].sum()),
                    "matched_solver_time_seconds": float(attempted["solver_time_seconds"].sum()),
                    "matched_total_time_seconds": float(
                        attempted["candidate_time_seconds"].sum()
                        + attempted["solver_time_seconds"].sum()
                    ),
                    "attempted_budgets": ",".join(str(int(value)) for value in attempted["budget"]),
                    "reporting_budget": endpoint,
                }
            )
            selected.append(row)

        reduced = pd.DataFrame(selected)
        if len(reduced) != 1800:
            raise ValueError(f"{dataset}/{method_label}: expected 1800 positives, found {len(reduced)}")
        if set(reduced["seed"].astype(int)) != {20260607, 20260608}:
            raise ValueError(f"{dataset}/{method_label}: incorrect seeds")
        if set(reduced["query_type"].astype(str)) != EXPECTED_FAMILIES:
            raise ValueError(f"{dataset}/{method_label}: incorrect query families")
        if set(reduced["target_query_size"].astype(int)) != EXPECTED_SIZES:
            raise ValueError(f"{dataset}/{method_label}: incorrect target sizes")
        cell_counts = reduced.groupby(["seed", "query_type", "target_query_size"]).size()
        if set(cell_counts.astype(int)) != {50}:
            raise ValueError(f"{dataset}/{method_label}: expected 50 queries per seed/family/size cell")

        solved = _as_bool(reduced["solver_found"])
        timed_out = _as_bool(reduced["solver_timed_out"])
        expected_solved = EXPECTED_SOLVED[(dataset, method_label)]
        if int(solved.sum()) != expected_solved:
            raise ValueError(
                f"{dataset}/{method_label}: solved {int(solved.sum())}, expected {expected_solved}"
            )
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method_label,
                "reporting_budget": endpoint,
                "budget_semantics": "exhaustive ceiling" if method_label == "FilterAll" else "half partitions",
                "positive_queries": len(reduced),
                "positive_solved": int(solved.sum()),
                "positive_solve_rate_percent": 100.0 * float(solved.mean()),
                "timeouts": int(timed_out.sum()),
                "avg_peak_pruned_candidate_nodes": float(reduced["peak_pruned_candidate_nodes"].mean()),
                "p50_peak_pruned_candidate_nodes": float(reduced["peak_pruned_candidate_nodes"].median()),
                "avg_pruned_candidate_nodes": float(reduced["pruned_candidate_nodes"].mean()),
                "p50_pruned_candidate_nodes": float(reduced["pruned_candidate_nodes"].median()),
                "avg_total_time_seconds": float(reduced["matched_total_time_seconds"].mean()),
                "p50_total_time_seconds": float(reduced["matched_total_time_seconds"].median()),
                "avg_solver_time_ms": 1000.0 * float(reduced["matched_solver_time_seconds"].mean()),
                "avg_candidate_time_seconds": float(reduced["matched_candidate_time_seconds"].mean()),
                "avg_retrieval_time_ms": 1000.0 * float(reduced["retrieval_time_seconds"].mean()),
                "source_dir": results_dir.as_posix(),
            }
        )
        detail_rows.extend(reduced.to_dict("records"))

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def _slice_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in details.groupby(
        ["dataset", "method_label", "query_type", "target_query_size"], sort=True
    ):
        dataset, method, family, size = keys
        solved = _as_bool(group["solver_found"])
        timed_out = _as_bool(group["solver_timed_out"])
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "query_family": family,
                "query_size": int(size),
                "queries": len(group),
                "solved": int(solved.sum()),
                "solve_rate_percent": 100.0 * float(solved.mean()),
                "timeouts": int(timed_out.sum()),
                "avg_pruned_candidate_nodes": float(group["pruned_candidate_nodes"].mean()),
                "avg_peak_pruned_candidate_nodes": float(group["peak_pruned_candidate_nodes"].mean()),
                "avg_candidate_time_seconds": float(
                    group["matched_candidate_time_seconds"].mean()
                ),
                "avg_solver_time_ms": 1000.0
                * float(group["matched_solver_time_seconds"].mean()),
                "avg_cascade_time_seconds": float(
                    group["matched_total_time_seconds"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cora-results",
        type=Path,
        default=Path("runs/lightning_completion/matched_cost_cora_v2/results"),
    )
    parser.add_argument(
        "--arxiv-results",
        type=Path,
        default=Path("runs/lightning_completion/matched_cost_arxiv_v2/results"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/paper_results/final_results/production_matched_costs.csv"),
    )
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=Path("runs/diagnostics/production_matched_costs_per_query.csv"),
    )
    parser.add_argument(
        "--slice-output",
        type=Path,
        default=Path(
            "benchmarks/paper_results/final_results/production_matched_costs_by_family_size.csv"
        ),
    )
    args = parser.parse_args()

    summaries = []
    details = []
    for dataset, results, half_budget, full_budget in (
        ("Cora", args.cora_results, 10, 20),
        ("Arxiv", args.arxiv_results, 100, 200),
    ):
        summary, detail = _reduce_dataset(dataset, results, half_budget, full_budget)
        summaries.append(summary)
        details.append(detail)

    output = args.output
    detail_output = args.detail_output
    output.parent.mkdir(parents=True, exist_ok=True)
    detail_output.parent.mkdir(parents=True, exist_ok=True)
    combined_details = pd.concat(details, ignore_index=True)
    pd.concat(summaries, ignore_index=True).to_csv(output, index=False)
    combined_details.to_csv(detail_output, index=False)
    args.slice_output.parent.mkdir(parents=True, exist_ok=True)
    _slice_summary(combined_details).to_csv(args.slice_output, index=False)
    print(f"wrote {output}")
    print(f"wrote {detail_output}")
    print(f"wrote {args.slice_output}")


if __name__ == "__main__":
    main()
