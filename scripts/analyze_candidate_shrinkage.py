"""Candidate-shrinkage cascade analysis.

Answers the central conceptual question about the Jigsaw overlap operator:
"If one-hop overlap can pull in the whole graph, are partitions acting as a tight
search boundary, or are we just expanding to (nearly) everything and relying on
pruning?"

For each method/query-family it characterizes the per-query cascade actually sent
toward the exact solver:

    full graph
      -> overlap_candidate_nodes      (selected partitions + one-hop overlap)
      -> signature_candidate_nodes    (after exact signature pruning)
      -> pruned_candidate_nodes       (after query-label-token pruning)
      -> component_solver_nodes        (after connected-component + label-cover filter)

It reports, per (dataset, method) and per (dataset, method, query_type, size):
  * absolute node counts at each cascade stage (mean / p50 / p95)
  * each stage as a fraction of the full graph
  * how often overlap is effectively the whole graph
  * how much pruning shrinks overlap (overlap / pruned, full / pruned)
  * node-coverage (FullCov) survival at overlap and after pruning
  * positive solve rate and stage wall-clock

Inputs are the same non-partial *_per_query.csv files the production summarizer
consumes. Per query, the "decisive" cascade row is the budget at which the query
first solved, else the largest attempted budget -- i.e. the candidate set that the
exact solver actually had to chew on. Use --budget to pin a fixed budget instead.

Outputs:
  runs/diagnostics/candidate_shrinkage_summary.csv      (dataset x method)
  runs/diagnostics/candidate_shrinkage_by_family.csv    (dataset x method x type x size)
  paper/fig_candidate_shrinkage_mag.png                 (optional, --figure)
"""

import argparse
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path


# --- value coercion (mirrors summarize_production_benchmarks.py) -------------

def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else 0.0


def percentile(values, p):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return 0.0
    k = (len(values) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def rate(flags):
    flags = list(flags)
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


# --- input discovery ---------------------------------------------------------

def infer_dataset(path, rows):
    for row in rows:
        value = row.get("dataset") or row.get("dataset_guess")
        if value:
            return value
    name = Path(path).name.lower()
    full = as_int(rows[0].get("full_graph_nodes")) if rows else 0
    if "mag" in name or full > 1_000_000:
        return "mag"
    if "arxiv" in name or full > 100_000:
        return "arxiv"
    if "cora" in name or (0 < full < 60_000):
        return "cora"
    parts = Path(path).name.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def is_partial_per_query_path(path):
    return "_partial_per_query" in Path(path).name


def iter_input_paths(patterns, include_partials=False):
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        for resolved in (matches if matches else [pattern]):
            if is_partial_per_query_path(resolved) and not include_partials:
                print(f"[SKIP PARTIAL] {resolved}", file=sys.stderr)
                continue
            real = str(Path(resolved).resolve())
            if real in seen:
                continue
            seen.add(real)
            yield resolved


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- per-query decisive-row selection ----------------------------------------

STAGES = [
    ("overlap", "overlap_candidate_nodes"),
    ("signature", "signature_candidate_nodes"),
    ("pruned", "pruned_candidate_nodes"),
    ("component", "component_solver_nodes"),
]


def decisive_row(qrows, pin_budget=None):
    """Pick the cascade row whose candidate set the solver actually faced.

    Default: the budget at which the query first solved; else the largest budget
    attempted. With pin_budget set, restrict to that exact budget.
    """
    if pin_budget is not None:
        pinned = [r for r in qrows if as_int(r.get("budget")) == pin_budget]
        if not pinned:
            return None
        qrows = pinned
    solved = [
        r for r in qrows
        if as_bool(r.get("cascade_first_solved")) and as_int(r.get("budget")) > 0
    ]
    if solved:
        return min(solved, key=lambda r: as_int(r.get("budget")))
    return max(qrows, key=lambda r: as_int(r.get("budget")))


def build_records(rows, dataset, include_negatives, pin_budget):
    """One decisive record per (model, query_id) in this file."""
    by_query = defaultdict(list)
    for row in rows:
        by_query[(row.get("model", ""), row.get("query_id", ""))].append(row)

    records = []
    for (_model, _qid), qrows in by_query.items():
        if not include_negatives and not as_bool(qrows[0].get("expected_match", "true")):
            continue
        row = decisive_row(qrows, pin_budget)
        if row is None:
            continue
        full = as_int(row.get("full_graph_nodes"))
        if full <= 0:
            continue
        stage_nodes = {name: as_float(row.get(col)) for name, col in STAGES}
        # component_solver_nodes is only populated when component_solve is on; if
        # it is 0 but the query was solved, the solver used the pruned set directly.
        if stage_nodes["component"] <= 0:
            stage_nodes["component"] = stage_nodes["pruned"]
        overlap = stage_nodes["overlap"]
        pruned = stage_nodes["pruned"]
        records.append({
            "dataset": dataset,
            "method": row.get("method", ""),
            "query_type": row.get("query_type", ""),
            "size": as_int(row.get("target_query_size")),
            "full": full,
            "budget": as_int(row.get("budget")),
            "nodes": stage_nodes,
            "frac": {k: (v / full if full else 0.0) for k, v in stage_nodes.items()},
            "overlap_fullcov": as_bool(row.get("overlap_node_fullcov")),
            "pruned_fullcov": as_bool(row.get("pruned_node_fullcov")),
            "solved": as_bool(row.get("cascade_first_solved")),
            "overlap_to_pruned": (overlap / pruned) if pruned > 0 else 0.0,
            "full_to_pruned": (full / pruned) if pruned > 0 else 0.0,
            "retrieval_s": as_float(row.get("retrieval_time_seconds")),
            "candidate_s": as_float(row.get("candidate_time_seconds")),
            "solver_s": as_float(row.get("solver_time_seconds")),
        })
    return records


# --- aggregation -------------------------------------------------------------

def summarize(records, group_keys, whole_graph_threshold):
    groups = defaultdict(list)
    for rec in records:
        groups[tuple(rec[k] for k in group_keys)].append(rec)

    out = []
    for key, recs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = dict(zip(group_keys, key))
        row["queries"] = len(recs)
        row["solve_rate"] = round(rate(r["solved"] for r in recs), 4)
        row["overlap_fullcov_rate"] = round(rate(r["overlap_fullcov"] for r in recs), 4)
        row["pruned_fullcov_rate"] = round(rate(r["pruned_fullcov"] for r in recs), 4)
        row["whole_graph_overlap_rate"] = round(
            rate(r["frac"]["overlap"] >= whole_graph_threshold for r in recs), 4
        )
        for name, _col in STAGES:
            row[f"mean_{name}_nodes"] = round(mean(r["nodes"][name] for r in recs), 1)
            row[f"p50_{name}_nodes"] = round(percentile([r["nodes"][name] for r in recs], 0.50), 1)
            row[f"p95_{name}_nodes"] = round(percentile([r["nodes"][name] for r in recs], 0.95), 1)
            row[f"mean_{name}_frac"] = round(mean(r["frac"][name] for r in recs), 4)
            row[f"p50_{name}_frac"] = round(percentile([r["frac"][name] for r in recs], 0.50), 4)
        row["mean_overlap_to_pruned"] = round(mean(r["overlap_to_pruned"] for r in recs), 2)
        row["p50_overlap_to_pruned"] = round(percentile([r["overlap_to_pruned"] for r in recs], 0.50), 2)
        row["mean_full_to_pruned"] = round(mean(r["full_to_pruned"] for r in recs), 2)
        row["mean_candidate_s"] = round(mean(r["candidate_s"] for r in recs), 3)
        row["mean_solver_s"] = round(mean(r["solver_s"] for r in recs), 3)
        row["mean_retrieval_s"] = round(mean(r["retrieval_s"] for r in recs), 3)
        out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        print(f"[WARN] no rows to write for {path}", file=sys.stderr)
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


# --- figure ------------------------------------------------------------------

def make_figure(summary_rows, dataset, out_path):
    rows = [r for r in summary_rows if r.get("dataset") == dataset]
    if not rows:
        print(f"[WARN] no {dataset} rows for figure; skipping", file=sys.stderr)
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] matplotlib unavailable, skipping figure: {exc}", file=sys.stderr)
        return

    method_order = [
        "hybrid", "coarse_mean_rrf", "mean_feature", "topo_feature", "random", "all",
    ]
    display_names = {
        "hybrid": "Jigsaw",
        "coarse_mean_rrf": "Mean-RRF",
        "mean_feature": "MeanFeat",
        "topo_feature": "TopoFeat",
        "random": "Random",
        "all": "FilterAll",
    }
    colors = {
        "hybrid": "#1f77b4",
        "coarse_mean_rrf": "#2ca02c",
        "mean_feature": "#9467bd",
        "topo_feature": "#8c564b",
        "random": "#7f7f7f",
        "all": "#d62728",
    }
    rows.sort(key=lambda r: (method_order.index(r["method"]) if r["method"] in method_order else 99, r["method"]))
    stage_labels = ["overlap", "signature", "pruned", "component"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(stage_labels))
    for r in rows:
        ys = [max(r[f"p50_{s}_nodes"], 1.0) for s in stage_labels]
        method = r["method"]
        label = display_names.get(method, method.replace("_", " ").title())
        ax.plot(
            list(x),
            ys,
            marker="o",
            color=colors.get(method),
            label=f"{label} (solve {r['solve_rate']*100:.0f}%)",
        )
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["overlap", "signature", "label-pruned", "solver\ncomponents"])
    ax.set_ylabel("median candidate nodes (log scale)")
    full = rows[0].get("p50_overlap_nodes", 0)
    ax.set_title(f"{dataset.upper()} candidate-shrinkage cascade (median nodes per query)")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2, framealpha=0.96)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# --- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+", help="per-query CSV files or globs")
    parser.add_argument("--summary-out", default="runs/diagnostics/candidate_shrinkage_summary.csv")
    parser.add_argument("--family-out", default="runs/diagnostics/candidate_shrinkage_by_family.csv")
    parser.add_argument("--figure", default="", help="path for the cascade figure (e.g. paper/fig_candidate_shrinkage_mag.png)")
    parser.add_argument("--figure-dataset", default="mag")
    parser.add_argument("--budget", type=int, default=None, help="pin a fixed budget instead of the decisive (first-solve/max) budget")
    parser.add_argument("--whole-graph-threshold", type=float, default=0.9, help="overlap frac >= this counts as 'whole-graph overlap'")
    parser.add_argument("--include-negatives", action="store_true", help="include negative queries (default: positives only)")
    parser.add_argument("--include-partials", action="store_true")
    args = parser.parse_args()

    records = []
    files = 0
    for path in iter_input_paths(args.csvs, include_partials=args.include_partials):
        rows = load_rows(path)
        if not rows:
            continue
        files += 1
        dataset = infer_dataset(path, rows)
        records.extend(build_records(rows, dataset, args.include_negatives, args.budget))

    if not records:
        print("No records produced. Check input paths / columns.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {files} files, {len(records)} decisive per-query records "
          f"({'positives+negatives' if args.include_negatives else 'positives only'}).")

    summary = summarize(records, ["dataset", "method"], args.whole_graph_threshold)
    family = summarize(records, ["dataset", "method", "query_type", "size"], args.whole_graph_threshold)

    write_csv(args.summary_out, summary)
    write_csv(args.family_out, family)

    if args.figure:
        make_figure(summary, args.figure_dataset, args.figure)

    # Console digest: the headline answer to "are partitions a tight boundary?"
    print("\n=== Candidate-shrinkage digest (median nodes; frac of full graph) ===")
    hdr = f"{'dataset':7} {'method':14} {'solve':>6} {'overlap':>16} {'pruned':>16} {'wholeG%':>8} {'O/P':>7}"
    print(hdr)
    for r in summary:
        print(
            f"{r['dataset']:7} {r['method']:14} {r['solve_rate']*100:5.1f}% "
            f"{r['p50_overlap_nodes']:>9,.0f} ({r['p50_overlap_frac']*100:4.0f}%) "
            f"{r['p50_pruned_nodes']:>9,.0f} ({r['p50_pruned_frac']*100:4.0f}%) "
            f"{r['whole_graph_overlap_rate']*100:7.0f}% "
            f"{r['p50_overlap_to_pruned']:>6.1f}x"
        )


if __name__ == "__main__":
    main()
