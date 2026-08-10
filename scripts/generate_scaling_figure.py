#!/usr/bin/env python
"""Generate paper/fig_scaling_feasibility.png from the paired direct-solver and
Jigsaw-cascade benchmark CSVs.

The figure tells the scaling story: the direct full-graph exact solver is feasible
on small graphs (Cora) but already infeasible at Arxiv scale (169K nodes) and at MAG
scale (1.9M); Jigsaw's retrieval-constrained cascade keeps the solver feasible across
all three. Two panels: (A) solve rate vs graph size, (B) per-query latency vs size.

Data sources (run with `python scripts/generate_scaling_figure.py`):
  Direct solver  : runs/directsolver_{cora,arxiv}_v2/results/*_per_query.csv  (method=fullgraph)
  Jigsaw cascade : deployed overlap-aware q50 runs, restricted to the same first
                   five K-hop queries at each size and each graph's half budget
  MAG            : production matrix (constants below; direct 0/15 timeout-censored,
                   cascade 88.6% ~6.97s). The retained MAG diagnostic does not preserve
                   a comparable wall-clock censoring point, so panel B omits that point.
"""
from __future__ import annotations
import glob
import math
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (nodes, dir-suffix). Cora/Arxiv read from CSV; MAG from production-matrix constants.
DATASETS = [
    ("Cora", 19_793),
    ("Arxiv", 169_343),
    ("MAG", 1_939_743),
]

# MAG production-matrix numbers (see paper Sec. 5; runs/directsolver_scaling_summary.md).
MAG_DIRECT = {"solve_rate": 0.0, "latency_s": float("nan"), "timeout": True}
MAG_CASCADE = {"solve_rate": 0.886, "latency_s": 6.97}  # walk-aware headline, matches Table production_matrix (avg_total_s)


def _per_query(path_glob: str) -> pd.DataFrame:
    files = [f for f in glob.glob(path_glob) if "partial" not in f]
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[0])


def _stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    solved = df["solver_found"].astype(str).str.lower().isin(["true", "1"]).mean()
    total = (
        df["retrieval_time_seconds"].fillna(0)
        + df["candidate_time_seconds"].fillna(0)
        + df["solver_time_seconds"].fillna(0)
    )
    timed_out = df["solver_timed_out"].astype(str).str.lower().isin(["true", "1"]).any()
    return {
        "solve_rate": float(solved),
        "latency_s": float(total.median()),
        "timeout": bool(timed_out and solved == 0),
    }


def _cascade_stats(path_glob: str, reporting_budget: int) -> dict:
    """Reduce a stopped cascade to the same 15 queries at a fixed budget."""
    df = _per_query(path_glob)
    if df.empty:
        return {}
    query_index = pd.to_numeric(df["query_id"].str.extract(r"_(\d+)$")[0])
    df = df.loc[
        df["query_type"].eq("k_hop")
        & df["target_query_size"].isin([20, 50, 100])
        & query_index.lt(5)
        & df["budget"].le(reporting_budget)
    ].copy()

    selected = []
    for _, group in df.groupby(["query_type", "target_query_size", "query_id"], sort=False):
        group = group.sort_values("budget")
        solved = group.loc[group["solver_found"].astype(str).str.lower().eq("true")]
        selected.append(solved.iloc[0] if not solved.empty else group.iloc[-1])
    return _stats(pd.DataFrame(selected))


def collect() -> tuple[list, list]:
    direct, cascade = [], []
    for name, nodes in DATASETS:
        if name == "MAG":
            direct.append((name, nodes, MAG_DIRECT))
            cascade.append((name, nodes, MAG_CASCADE))
            continue
        ds = name.lower()
        d = _stats(_per_query(f"{ROOT}/runs/directsolver_{ds}_v2/results/*_per_query.csv"))
        reporting_budget = {"cora": 10, "arxiv": 100}[ds]
        c = _cascade_stats(
            f"{ROOT}/runs/lightning_completion/jigsaw-{ds}-overlap-graphsage-bench-gcp-cpux8-v3/"
            f"results/*s20260607*neural*per_query.csv",
            reporting_budget,
        )
        direct.append((name, nodes, d))
        cascade.append((name, nodes, c))
    return direct, cascade


def main() -> None:
    direct, cascade = collect()
    names = [d[0] for d in direct]
    nodes = [d[1] for d in direct]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # ---- Panel A: solve rate ----
    d_solve = [100 * d[2].get("solve_rate", 0) for d in direct]
    c_solve = [100 * c[2].get("solve_rate", 0) for c in cascade]
    x = range(len(names))
    w = 0.38
    axL.bar(
        [i - w / 2 for i in x], d_solve, w,
        color="#c44e52", edgecolor="#333",
    )
    axL.bar(
        [i + w / 2 for i in x], c_solve, w,
        color="#4c72b0", edgecolor="#333",
    )
    for i, v in enumerate(d_solve):
        axL.text(i - w / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=8)
    for i, v in enumerate(c_solve):
        axL.text(i + w / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=8)
    axL.set_xticks(list(x))
    axL.set_xticklabels([f"{n}\n({k/1000:.0f}K nodes)" if k < 1e6 else f"{n}\n({k/1e6:.1f}M nodes)"
                         for n, k in zip(names, nodes)])
    axL.set_ylabel("Queries solved exactly (%)")
    axL.set_ylim(0, 112)
    axL.set_title("(a) Feasibility vs graph scale")
    axL.axhline(0, color="#888", lw=0.6)

    # ---- Panel B: latency (log-log) ----
    d_lat = [d[2].get("latency_s", float("nan")) for d in direct]
    c_lat = [c[2].get("latency_s", float("nan")) for c in cascade]
    d_to = [d[2].get("timeout", False) for d in direct]
    # Solid for solved points, dashed for censored timeouts (solver killed, not a real runtime).
    axR.plot(nodes, d_lat, "--", color="#c44e52", alpha=0.5, zorder=1)
    solved_x = [n for n, l, to in zip(nodes, d_lat, d_to) if not to and math.isfinite(l)]
    solved_y = [l for l, to in zip(d_lat, d_to) if not to and math.isfinite(l)]
    to_x = [n for n, l, to in zip(nodes, d_lat, d_to) if to and math.isfinite(l)]
    to_y = [l for l, to in zip(d_lat, d_to) if to and math.isfinite(l)]
    solved_handle, = axR.plot(solved_x, solved_y, "o", color="#c44e52")
    timeout_handle, = axR.plot(
        to_x, to_y, "x", color="#c44e52", markersize=9, mew=2,
    )
    cascade_handle, = axR.plot(nodes, c_lat, "s-", color="#4c72b0")
    for n, lat in zip(to_x, to_y):
        axR.annotate(
            "timeout",
            (n, lat),
            textcoords="offset points",
            xytext=(-8, -6),
            ha="right",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="#c44e52",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.0},
        )
    axR.set_xscale("log")
    axR.set_yscale("log")
    finite_latencies = [
        value for value in d_lat + c_lat if math.isfinite(value) and value > 0
    ]
    axR.set_ylim(min(finite_latencies) * 0.65, max(finite_latencies) * 2.2)
    axR.set_xlabel("Graph size (nodes, log)")
    axR.set_ylabel("Median per-query time (s, log)")
    axR.set_title("(b) Per-query latency")
    axR.legend(
        [solved_handle, timeout_handle, cascade_handle],
        ["Direct full-graph solver (solved)", "Direct solver (timeout, 0 solved)", "Jigsaw (cascade)"],
        loc="upper right",
        fontsize=8,
        framealpha=0.96,
    )
    axR.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "fig_scaling_feasibility.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    summary_rows = []
    for (name, nodes, direct_stats), (_, _, cascade_stats) in zip(direct, cascade):
        summary_rows.append(
            {
                "dataset": name.lower(),
                "nodes": nodes,
                "paired_queries": 15 if name != "MAG" else "separate diagnostics",
                "jigsaw_budget": {"Cora": 10, "Arxiv": 100, "MAG": 1000}[name],
                "jigsaw_budget_fraction": 0.5,
                "direct_solve_rate_percent": 100 * direct_stats.get("solve_rate", 0),
                "direct_median_seconds": direct_stats.get("latency_s"),
                "jigsaw_solve_rate_percent": 100 * cascade_stats.get("solve_rate", 0),
                "jigsaw_median_seconds": cascade_stats.get("latency_s"),
            }
        )
    summary_path = os.path.join(
        ROOT, "benchmarks", "paper_results", "ablations", "scaling_half_budget_paired_summary.csv"
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print("wrote", out)
    print("wrote", summary_path)
    print("direct :", [(d[0], round(100 * d[2].get("solve_rate", 0)), round(d[2].get("latency_s", float('nan')), 2)) for d in direct])
    print("cascade:", [(c[0], round(100 * c[2].get("solve_rate", 0)), round(c[2].get("latency_s", float('nan')), 2)) for c in cascade])


if __name__ == "__main__":
    main()
