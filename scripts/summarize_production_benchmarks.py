import argparse
import csv
import glob
import math
import re
import sys
from pathlib import Path


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except ValueError:
        return default


def as_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def percentile(values, p):
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return 0.0
    k = (len(values) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_dataset(path, rows):
    for row in rows:
        value = row.get("dataset") or row.get("dataset_guess")
        if value:
            return value
    name = Path(path).name.lower()
    if "mag" in name:
        return "mag"
    if "arxiv" in name:
        return "arxiv"
    if "cora" in name:
        return "cora"
    parts = Path(path).name.split("_")
    return parts[1] if len(parts) > 1 else ""


def infer_seed(path, rows):
    for row in rows:
        for key in ("seed", "query_seed"):
            value = row.get(key, "")
            if re.fullmatch(r"20\d{6}", str(value)):
                return str(value)
    text = " ".join([str(path), *(str(row.get("file", "")) for row in rows[:3])])
    match = re.search(r"(?:^|[_\\/.-])s(20\d{6})(?:[_\\/.-]|$)", text)
    return match.group(1) if match else ""


def summarize_group(path, rows, group_key):
    by_query = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)

    solved_budget = {}
    expected_match = {}
    contained = {}
    timed_out = {}
    totals = []
    candidates = []
    component_nodes = []
    reduction = []
    retrieval = []
    candidate_time = []
    solver_time = []
    true_parts = []
    precision = []
    recall = []
    max_rank = []
    edge_reduction = []

    for query_id, qrows in by_query.items():
        solved = [row for row in qrows if as_bool(row.get("cascade_first_solved"))]
        solved_budgets = [as_int(row.get("budget")) for row in solved]
        solved_budgets = [budget for budget in solved_budgets if budget > 0]
        solved_budget[query_id] = min(solved_budgets) if solved_budgets else 0
        expected_match[query_id] = as_bool(qrows[0].get("expected_match", "true"))
        contained[query_id] = any(as_bool(row.get("pruned_node_fullcov")) for row in qrows)
        timed_out[query_id] = any(as_bool(row.get("solver_timed_out")) for row in qrows)
        totals.append(
            max(
                as_float(row.get("cascade_total_candidate_time_seconds"))
                + as_float(row.get("cascade_total_solver_time_seconds"))
                for row in qrows
            )
        )
        candidates.append(max(as_float(row.get("pruned_candidate_nodes")) for row in qrows))
        component_nodes.append(max(as_float(row.get("component_solver_nodes")) for row in qrows))
        reduction.append(max(as_float(row.get("node_reduction_factor")) for row in qrows))
        retrieval.extend(as_float(row.get("retrieval_time_seconds")) for row in qrows)
        candidate_time.extend(as_float(row.get("candidate_time_seconds")) for row in qrows)
        solver_time.extend(as_float(row.get("solver_time_seconds")) for row in qrows)
        true_parts.append(max(as_float(row.get("true_coarse_count")) for row in qrows))
        precision.extend(as_float(row.get("coarse_precision_at_budget")) for row in qrows)
        recall.extend(as_float(row.get("coarse_recall_at_budget")) for row in qrows)
        max_rank.append(max(as_float(row.get("max_true_coarse_rank")) for row in qrows))
        edge_reduction.extend(
            as_float(row.get("edge_reduction_factor"))
            for row in qrows
            if as_float(row.get("edge_reduction_factor")) > 0
        )

    n = len(by_query)
    positive_ids = [qid for qid in by_query if expected_match.get(qid, True)]
    negative_ids = [qid for qid in by_query if not expected_match.get(qid, True)]
    budgets = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    filename = Path(path).name
    label_match = re.search(r"_sizes(?:\d+_?)+_(.+?)_b(?:\d|$)", filename)
    method_label = label_match.group(1) if label_match else group_key[3]
    out = {
        "file": str(path),
        "dataset": infer_dataset(path, rows),
        "dataset_guess": infer_dataset(path, rows),
        "seed": infer_seed(path, rows),
        "model": group_key[0],
        "query_type": group_key[1],
        "target_query_size": group_key[2],
        "method": group_key[3],
        "method_label": method_label,
        "signature": group_key[4],
        "queries": n,
        "solved": sum(1 for b in solved_budget.values() if b > 0),
        "positive_queries": len(positive_ids),
        "negative_queries": len(negative_ids),
        "positive_solved": sum(1 for qid in positive_ids if solved_budget.get(qid, 0) > 0),
        "false_positives": sum(1 for qid in negative_ids if solved_budget.get(qid, 0) > 0),
        "correct_no_match": sum(
            1
            for qid in negative_ids
            if solved_budget.get(qid, 0) == 0 and not timed_out.get(qid, False)
        ),
        "negative_timeouts": sum(1 for qid in negative_ids if timed_out.get(qid, False)),
        "unknown_within_budget": sum(
            1
            for qid in by_query
            if solved_budget.get(qid, 0) == 0 and timed_out.get(qid, False)
        ),
        "contained": sum(1 for v in contained.values() if v),
        "timeouts": sum(1 for v in timed_out.values() if v),
        "avg_total_s": sum(totals) / n if n else 0.0,
        "p50_total_s": percentile(totals, 0.50),
        "p95_total_s": percentile(totals, 0.95),
        "avg_candidate_nodes": sum(candidates) / n if n else 0.0,
        "p95_candidate_nodes": percentile(candidates, 0.95),
        "avg_component_solver_nodes": sum(component_nodes) / n if n else 0.0,
        "avg_node_reduction_factor": sum(reduction) / n if n else 0.0,
        "avg_edge_reduction_factor_diag": sum(edge_reduction) / len(edge_reduction) if edge_reduction else 0.0,
        "avg_true_parts": sum(true_parts) / n if n else 0.0,
        "avg_precision_at_budget": sum(precision) / len(precision) if precision else 0.0,
        "avg_recall_at_budget": sum(recall) / len(recall) if recall else 0.0,
        "p50_max_true_rank": percentile(max_rank, 0.50),
        "p95_max_true_rank": percentile(max_rank, 0.95),
        "p99_max_true_rank": percentile(max_rank, 0.99),
        "avg_retrieval_ms": 1000.0 * sum(retrieval) / len(retrieval) if retrieval else 0.0,
        "avg_candidate_ms": 1000.0 * sum(candidate_time) / len(candidate_time) if candidate_time else 0.0,
        "avg_solver_ms": 1000.0 * sum(solver_time) / len(solver_time) if solver_time else 0.0,
    }
    running_solved = 0
    for budget in budgets:
        exact = sum(1 for b in solved_budget.values() if b == budget)
        running_solved += exact
        # Backward-compatible exact first-hit bucket. Prefer the explicit aliases
        # below in new analysis code and paper tables.
        out[f"solved_at_{budget}"] = exact
        out[f"first_solved_at_{budget}"] = exact
        out[f"solved_by_{budget}"] = running_solved
    return out


def summarize_file(path):
    rows = load_rows(path)
    groups = {}
    for row in rows:
        key = (
            row.get("model", ""),
            row.get("query_type", "k_hop"),
            row.get("target_query_size", ""),
            row.get("method", ""),
            row.get("signature", ""),
        )
        groups.setdefault(key, []).append(row)
    return [summarize_group(path, group_rows, key) for key, group_rows in sorted(groups.items())]


def is_partial_per_query_path(path):
    return "_partial_per_query" in Path(path).name


def iter_input_paths(patterns, include_partials=False):
    for path in patterns:
        matches = sorted(glob.glob(path))
        for resolved in (matches if matches else [path]):
            if is_partial_per_query_path(resolved) and not include_partials:
                print(f"[SKIP PARTIAL] {resolved}", file=sys.stderr)
                continue
            yield resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--include-partials",
        action="store_true",
        help="Include rolling *_partial_per_query.csv files. Off by default for canonical summaries.",
    )
    args = parser.parse_args()

    rows = []
    for resolved in iter_input_paths(args.csvs, include_partials=args.include_partials):
        rows.extend(summarize_file(resolved))
    fieldnames = list(rows[0].keys()) if rows else []
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
