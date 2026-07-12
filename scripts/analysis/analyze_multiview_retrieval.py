"""Aggregate paired fixed-versus-multi-view retrieval results."""

import argparse
import csv
import math
from collections import defaultdict


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def exact_mcnemar_p(wins, losses):
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(0, min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    args = parser.parse_args()

    rows = []
    for path in args.csv:
        with open(path, newline="", encoding="utf-8") as handle:
            seed = path.rsplit("seed", 1)[-1].split("_", 1)[0]
            for row in csv.DictReader(handle):
                row["_seed"] = seed
                rows.append(row)

    grouped = defaultdict(dict)
    recalls = defaultdict(list)
    for row in rows:
        key = (
            row["_seed"],
            row["model"],
            row["query_id"],
            int(row["coarse_budget"]),
        )
        grouped[key][row["method"]] = as_bool(row["expanded_fullcov"])
        recalls[(row["method"], int(row["coarse_budget"]))].append(
            float(row["expanded_recall"])
        )

    methods = sorted(
        {
            method
            for method_map in grouped.values()
            for method in method_map
            if method != "fixed"
        }
    )
    budgets = sorted({key[-1] for key in grouped})

    print(
        "| Method | Budget | FullCov | Fixed FullCov | Delta | "
        "Wins | Losses | Ties | Exact McNemar p | Avg Recall |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for method in methods:
        for budget in budgets:
            pairs = [
                method_map
                for key, method_map in grouped.items()
                if key[-1] == budget and "fixed" in method_map and method in method_map
            ]
            fixed = sum(pair["fixed"] for pair in pairs)
            candidate = sum(pair[method] for pair in pairs)
            wins = sum(pair[method] and not pair["fixed"] for pair in pairs)
            losses = sum(pair["fixed"] and not pair[method] for pair in pairs)
            ties = len(pairs) - wins - losses
            avg_recall = sum(recalls[(method, budget)]) / len(recalls[(method, budget)])
            print(
                f"| {method} | {budget} | {candidate}/{len(pairs)} | "
                f"{fixed}/{len(pairs)} | {candidate - fixed:+d} | {wins} | "
                f"{losses} | {ties} | {exact_mcnemar_p(wins, losses):.6f} | "
                f"{avg_recall:.4f} |"
            )


if __name__ == "__main__":
    main()
