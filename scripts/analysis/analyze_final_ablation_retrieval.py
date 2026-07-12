"""Aggregate the locked two-seed matched-budget retrieval evaluation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


LOCKED_METHOD = "coarse_hybrid_mw0.5_teleport10"
METHODS = ("fixed", LOCKED_METHOD)
BUDGETS = {
    "fixed": (20, 50, 100),
    LOCKED_METHOD: (50, 75, 100),
}


def variant_name(model: str) -> str:
    for variant in ("control", "cvar", "topk", "final"):
        if f"fair_{variant}_" in model:
            return variant
    return "other"


def checkpoint_type(model: str) -> str:
    return "best" if model.endswith("_best") else "final"


def aggregate_summary(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["query_seed"] = path.stem.split("_seed")[-1].split("_")[0]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data[data["method"].isin(METHODS)].copy()
    data = data[
        data.apply(
            lambda row: int(row["coarse_budget"]) in BUDGETS[row["method"]], axis=1
        )
    ].copy()
    data["recall_weight"] = data["avg_expanded_recall"] * data["queries"]
    data["rank_weight"] = data["avg_max_true_coarse_rank"] * data["queries"]
    result = (
        data.groupby(["model", "method", "coarse_budget"], as_index=False)
        .agg(
            queries=("queries", "sum"),
            fullcov=("expanded_fullcov", "sum"),
            recall_weight=("recall_weight", "sum"),
            rank_weight=("rank_weight", "sum"),
        )
        .sort_values(["model", "method", "coarse_budget"])
    )
    result["fullcov_rate"] = result["fullcov"] / result["queries"]
    result["recall"] = result["recall_weight"] / result["queries"]
    result["avg_max_true_rank"] = result["rank_weight"] / result["queries"]
    result["variant"] = result["model"].map(variant_name)
    result["checkpoint"] = result["model"].map(checkpoint_type)
    return result.drop(columns=["recall_weight", "rank_weight"])


def impossible_counts(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        frame = pd.read_csv(
            path,
            usecols=["query_id", "true_coarse_count", "model", "method", "coarse_budget"],
        )
        queries = frame[
            (frame["model"] == frame["model"].iloc[0])
            & (frame["method"] == "fixed")
            & (frame["coarse_budget"] == 20)
        ][["query_id", "true_coarse_count"]].drop_duplicates()
        for budget in (20, 50, 100):
            rows.append(
                {
                    "query_seed": path.stem.split("_seed")[-1].split("_")[0],
                    "budget": budget,
                    "queries": len(queries),
                    "impossible": int((queries["true_coarse_count"] > budget).sum()),
                    "avg_true_coarse": queries["true_coarse_count"].mean(),
                    "max_true_coarse": queries["true_coarse_count"].max(),
                }
            )
    return pd.DataFrame(rows)


def exact_mcnemar_pvalue(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_control_final(paths: list[Path]) -> pd.DataFrame:
    rows = []
    pairs = (
        ("fair_control_seed7101_best", "fair_final_seed7101_best"),
        ("fair_control_seed7102_best", "fair_final_seed7102_best"),
    )
    for path in paths:
        frame = pd.read_csv(
            path,
            usecols=["model", "query_id", "method", "coarse_budget", "expanded_fullcov"],
        )
        for control, final in pairs:
            for method in METHODS:
                for budget in BUDGETS[method]:
                    selected = frame[
                        (frame["method"] == method)
                        & (frame["coarse_budget"] == budget)
                        & (frame["model"].isin([control, final]))
                    ]
                    pivot = selected.pivot(
                        index="query_id", columns="model", values="expanded_fullcov"
                    )
                    rows.append(
                        {
                            "method": method,
                            "budget": budget,
                            "wins": int(
                                ((pivot[final] == 1) & (pivot[control] == 0)).sum()
                            ),
                            "losses": int(
                                ((pivot[final] == 0) & (pivot[control] == 1)).sum()
                            ),
                        }
                    )
    result = (
        pd.DataFrame(rows)
        .groupby(["method", "budget"], as_index=False)[["wins", "losses"]]
        .sum()
    )
    result["exact_mcnemar_p"] = result.apply(
        lambda row: exact_mcnemar_pvalue(int(row["wins"]), int(row["losses"])),
        axis=1,
    )
    return result


def make_markdown(
    aggregate: pd.DataFrame, impossible: pd.DataFrame, paired: pd.DataFrame
) -> str:
    best = aggregate[aggregate["checkpoint"] == "best"].copy()
    variant = (
        best.groupby(["variant", "method", "coarse_budget"], as_index=False)
        .agg(
            training_replicates=("model", "nunique"),
            queries=("queries", "sum"),
            fullcov=("fullcov", "sum"),
            recall=("recall", "mean"),
            avg_max_true_rank=("avg_max_true_rank", "mean"),
        )
        .sort_values(["method", "coarse_budget", "variant"])
    )
    variant["fullcov_rate"] = variant["fullcov"] / variant["queries"]

    lines = [
        "# Fair Matched-Budget Arxiv Retrieval",
        "",
        "Two locked query seeds, 100 queries each. Every training run used exactly "
        "9,000 optimizer steps. FullCov is primary; recall is secondary.",
        "",
        "## Best-selected checkpoints by objective",
        "",
        "| Objective | Retrieval | Budget | Training replicates | FullCov | Rate | Recall | Avg max true rank |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in variant.itertuples():
        retrieval = "fixed" if row.method == "fixed" else "locked dynamic"
        lines.append(
            f"| {row.variant} | {retrieval} | {row.coarse_budget} | "
            f"{row.training_replicates} | {row.fullcov}/{row.queries} | "
            f"{row.fullcov_rate:.3f} | {row.recall:.4f} | "
            f"{row.avg_max_true_rank:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Individual final-objective checkpoints",
            "",
            "| Checkpoint | Retrieval | Budget | FullCov | Recall | Avg max true rank |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    final_rows = aggregate[aggregate["variant"] == "final"]
    for row in final_rows.itertuples():
        retrieval = "fixed" if row.method == "fixed" else "locked dynamic"
        lines.append(
            f"| {row.model} | {retrieval} | {row.coarse_budget} | "
            f"{row.fullcov}/{row.queries} | {row.recall:.4f} | "
            f"{row.avg_max_true_rank:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Paired control versus final FullCov",
            "",
            "Wins are queries covered only by final; losses are queries covered only "
            "by control. The exact two-sided McNemar p-value pools both training "
            "replicates and both locked query seeds.",
            "",
            "| Retrieval | Budget | Final wins | Final losses | Exact p-value |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in paired.itertuples():
        retrieval = "fixed" if row.method == "fixed" else "locked dynamic"
        lines.append(
            f"| {retrieval} | {row.budget} | {row.wins} | {row.losses} | "
            f"{row.exact_mcnemar_p:.4g} |"
        )

    pooled_impossible = (
        impossible.groupby("budget", as_index=False)
        .agg(
            queries=("queries", "sum"),
            impossible=("impossible", "sum"),
            avg_true_coarse=("avg_true_coarse", "mean"),
            max_true_coarse=("max_true_coarse", "max"),
        )
        .sort_values("budget")
    )
    lines.extend(
        [
            "",
            "## Query feasibility",
            "",
            "| K | Impossible queries | Avg true partitions | Max true partitions |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pooled_impossible.itertuples():
        lines.append(
            f"| {row.budget} | {row.impossible}/{row.queries} | "
            f"{row.avg_true_coarse:.2f} | {row.max_true_coarse} |"
        )
    lines.extend(
        [
            "",
            "Retrieval latency is not present in these CSVs and must not be claimed "
            "from this evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("runs/logs"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("runs/logs/fair_ablation_locked_aggregate"),
    )
    args = parser.parse_args()

    summary_paths = sorted(
        args.input_dir.glob("retrieval_arxiv_khop_fair_ablation_q100_seed*_summary.csv")
    )
    query_paths = sorted(
        args.input_dir.glob("retrieval_arxiv_khop_fair_ablation_q100_seed*_per_query.csv")
    )
    if len(summary_paths) != 2 or len(query_paths) != 2:
        raise SystemExit("Expected exactly two locked summary and two per-query CSVs.")

    aggregate = aggregate_summary(summary_paths)
    impossible = impossible_counts(query_paths)
    paired = paired_control_final(query_paths)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    impossible.to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_feasibility.csv"),
        index=False,
    )
    paired.to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_paired.csv"),
        index=False,
    )
    args.output_prefix.with_suffix(".md").write_text(
        make_markdown(aggregate, impossible, paired), encoding="utf-8"
    )
    print(args.output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
