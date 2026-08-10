#!/usr/bin/env python3
"""Fail fast when paper-facing comparisons mix retrieval budget fractions."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPERS = [
    ROOT / "paper" / "samplepaper.tex",
    ROOT / "paper" / "jigsaw_log2026.tex",
    ROOT / "paper" / "jigsaw_ecmlpkdd.tex",
]
PARTITIONS = {"cora": 20, "arxiv": 200, "mag": 2000}
HALF_BUDGETS = {name: count // 2 for name, count in PARTITIONS.items()}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    errors: list[str] = []

    operator_path = (
        ROOT
        / "benchmarks"
        / "paper_results"
        / "ablations"
        / "operator_ablation_half_budget_summary.csv"
    )
    operator_rows = read_csv(operator_path)
    expected_variants = {
        "full",
        "no_overlap",
        "no_signature",
        "no_components",
        "no_exact_label",
    }
    for dataset in ("arxiv", "mag"):
        rows = [row for row in operator_rows if row["dataset"] == dataset]
        if {row["variant"] for row in rows} != expected_variants:
            errors.append(f"operator variants are incomplete for {dataset}")
        for row in rows:
            if int(row["total_coarse_partitions"]) != PARTITIONS[dataset]:
                errors.append(f"wrong partition count in {operator_path}: {row}")
            if int(row["reporting_budget"]) != HALF_BUDGETS[dataset]:
                errors.append(f"operator ablation is not at half budget: {row}")
            if float(row["budget_fraction"]) != 0.5:
                errors.append(f"operator budget fraction is not 0.5: {row}")
            if int(row["positive_queries"]) != 36:
                errors.append(f"operator denominator is not 36 positives: {row}")

    scaling_path = (
        ROOT
        / "benchmarks"
        / "paper_results"
        / "ablations"
        / "scaling_half_budget_paired_summary.csv"
    )
    scaling_rows = {row["dataset"]: row for row in read_csv(scaling_path)}
    for dataset in ("cora", "arxiv"):
        row = scaling_rows.get(dataset)
        if row is None:
            errors.append(f"missing paired scaling row for {dataset}")
            continue
        if row["paired_queries"] != "15":
            errors.append(f"paired scaling denominator is not 15 for {dataset}")
        if int(row["jigsaw_budget"]) != HALF_BUDGETS[dataset]:
            errors.append(f"paired scaling is not at half budget for {dataset}")
        if float(row["jigsaw_budget_fraction"]) != 0.5:
            errors.append(f"paired scaling fraction is not 0.5 for {dataset}")

    manifest_path = ROOT / "benchmarks" / "canonical_workload_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics = manifest["diagnostic_workloads"]
    operator_manifest = diagnostics["operator_ablation"]
    scaling_manifest = diagnostics["paired_scaling"]
    if operator_manifest["reported_budget_fraction"] != 0.5:
        errors.append("operator manifest budget fraction is not 0.5")
    if scaling_manifest["reported_budget_fraction"] != 0.5:
        errors.append("scaling manifest budget fraction is not 0.5")
    for dataset in ("arxiv", "mag"):
        if operator_manifest["reported_budgets"][dataset] != HALF_BUDGETS[dataset]:
            errors.append(f"operator manifest budget is wrong for {dataset}")
    for dataset in ("cora", "arxiv"):
        if scaling_manifest["reported_budgets"][dataset] != HALF_BUDGETS[dataset]:
            errors.append(f"scaling manifest budget is wrong for {dataset}")

    stale_claims = [
        "overlap is a no-op on Arxiv",
        "all variants still solve 100",
        "7{,}081",
        "75\\times",
        "14/15",
        "100/93/89",
        "recall-preserving \\emph{selective overlap}",
        "Cora & MeanFeat & 88.8",
        "Cora & TopoFeat & 33.4",
        "Cora & Random & 31.3",
        "Arxiv & MeanFeat & 88.9",
        "Arxiv & TopoFeat & 33.2",
        "Arxiv & Random & 37.7",
        "cost cells are omitted",
        "cross-policy costs are omitted",
    ]
    required_claims = [
        "matched half-partition budget",
        "94.4{\\to}86.1",
        "2{,}815",
    ]

    matched_path = (
        ROOT / "benchmarks" / "paper_results" / "final_results" / "production_matched_costs.csv"
    )
    matched_rows = read_csv(matched_path)
    if len(matched_rows) != 12:
        errors.append(f"expected 12 matched production rows, found {len(matched_rows)}")

    def candidate_text(value: float) -> str:
        if value >= 1000:
            return f"{value / 1000.0:.2f}K"
        return str(int(round(value)))

    expected_lines: list[str] = []
    for row in matched_rows:
        dataset = row["dataset"]
        method = row["method"]
        budget = int(row["reporting_budget"])
        expected_budget = PARTITIONS[dataset.lower()] if method == "FilterAll" else HALF_BUDGETS[dataset.lower()]
        if budget != expected_budget:
            errors.append(f"wrong reporting budget in matched costs: {row}")
        if int(row["positive_queries"]) != 1800:
            errors.append(f"wrong positive denominator in matched costs: {row}")
        if int(row["timeouts"]) != 0:
            errors.append(f"unexpected Cora/Arxiv timeout in matched costs: {row}")
        method_tex = "FilterAll$^\\ddagger$" if method == "FilterAll" else method
        expected_lines.append(
            f"{dataset} & {method_tex} & "
            f"{float(row['positive_solve_rate_percent']):.1f} & 100.0 & 0 & 0 & "
            f"{candidate_text(float(row['avg_pruned_candidate_nodes']))} & "
            f"{float(row['avg_total_time_seconds']):.2f} & "
            f"{float(row['avg_solver_time_ms']):.1f} \\\\"
        )

    for paper in PAPERS:
        text = paper.read_text(encoding="utf-8")
        for stale in stale_claims:
            if stale.lower() in text.lower():
                errors.append(f"stale budget claim in {paper.name}: {stale}")
        for required in required_claims:
            if required not in text:
                errors.append(f"missing budget-fairness marker in {paper.name}: {required}")
        lines = text.splitlines()
        for expected in expected_lines:
            if lines.count(expected) != 1:
                errors.append(f"matched production row missing or duplicated in {paper.name}: {expected}")

    if errors:
        print("Paper budget fairness validation FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Paper budget fairness validation passed: operator and paired-scaling "
        "diagnostics use matched half-partition budgets, every Cora/Arxiv table "
        "row matches the canonical cost reducer, and stale claims are absent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
