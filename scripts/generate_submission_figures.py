from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "benchmarks" / "paper_results" / "final_results"
PAPER_DIR = ROOT / "paper"
DIAGNOSTIC_DIR = ROOT / "runs" / "diagnostics"

CONNECTED_SUMMARIES = {
    "cora": ROOT
    / "runs"
    / "lightning_connected_reruns"
    / "jigsaw-cora-mf-mc-connected-gcp-cpux8-v3"
    / "summary.csv",
    "arxiv": ROOT / "runs" / "lcr_arxiv_v3" / "summary.csv",
    "mag": ROOT / "runs" / "lcr_mag_v3" / "summary.csv",
}

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

DATASET_ORDER = ["cora", "arxiv", "mag"]
DATASET_LABEL = {"cora": "Cora", "arxiv": "Arxiv", "mag": "MAG"}
METHOD_ORDER = ["hybrid", "coarse_mean_rrf", "mean_feature", "topo_feature", "random", "all"]
METHOD_LABEL = {
    "hybrid": "Jigsaw",
    "coarse_mean_rrf": "Mean-RRF",
    "mean_feature": "MeanFeat",
    "topo_feature": "TopoFeat",
    "random": "Random",
    "all": "FilterAll",
}
METHOD_COLOR = {
    "Jigsaw": "#1f77b4",
    "Mean-RRF": "#2ca02c",
    "MeanFeat": "#9467bd",
    "TopoFeat": "#8c564b",
    "Random": "#7f7f7f",
    "FilterAll": "#d62728",
}

CONNECTED_QUERY_TYPES = {"multi_fine", "multi_coarse"}


def load_summary_for_figures() -> pd.DataFrame:
    """Load the canonical summary and replace corrected connected-query slices.

    The final CSV is intentionally conservative and only changes when the full
    repair pass is complete. Figures can still use already validated corrected
    connected reruns for Cora/Arxiv, and will automatically pick up MAG once its
    complete rerun summary lands at runs/lcr_mag_v3/summary.csv.
    """
    df = pd.read_csv(FINAL_DIR / "final_all_datasets_summary.csv", low_memory=False)
    schema = list(df.columns)
    replacements = []
    for dataset, path in CONNECTED_SUMMARIES.items():
        if not path.exists():
            continue
        replacement = pd.read_csv(path, low_memory=False)
        replacement = replacement[replacement["query_type"].isin(CONNECTED_QUERY_TYPES)].copy()
        if replacement.empty:
            continue
        if set(replacement["seed"].astype(str)) != {"20260607", "20260608"}:
            continue
        if replacement.groupby(["method", "seed", "query_type", "target_query_size"]).size().min() < 1:
            continue
        replacement["source_file"] = str(path)
        missing = [column for column in schema if column not in replacement.columns]
        if missing:
            replacement = pd.concat(
                [replacement, pd.DataFrame({column: np.nan for column in missing}, index=replacement.index)],
                axis=1,
            )
        replacement = replacement[schema].copy()
        df = df[
            ~(
                df["dataset"].eq(dataset)
                & df["query_type"].isin(CONNECTED_QUERY_TYPES)
            )
        ].copy()
        replacements.append(replacement)
    if replacements:
        df = pd.concat([df, *replacements], ignore_index=True)
    return df


def weighted_avg(group: pd.DataFrame, column: str) -> float:
    weights = pd.to_numeric(group["queries"], errors="coerce").fillna(0.0)
    values = pd.to_numeric(group[column], errors="coerce").fillna(0.0)
    return float((values * weights).sum() / weights.sum()) if weights.sum() else 0.0


def method_totals(df: pd.DataFrame, exclude_query_types: set[str] | None = None) -> pd.DataFrame:
    if exclude_query_types:
        df = df[~df["query_type"].isin(exclude_query_types)].copy()
    rows = []
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            group = df[(df["dataset"].eq(dataset)) & (df["method"].eq(method))]
            pos_q = pd.to_numeric(group["positive_queries"], errors="coerce").sum()
            neg_q = pd.to_numeric(group["negative_queries"], errors="coerce").sum()
            pos = pd.to_numeric(group["positive_solved"], errors="coerce").sum()
            no_match = pd.to_numeric(group["correct_no_match"], errors="coerce").sum()
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABEL[dataset],
                    "method": method,
                    "method_label": METHOD_LABEL[method],
                    "positive_solve_rate": 100.0 * pos / pos_q if pos_q else np.nan,
                    "correct_no_match_rate": 100.0 * no_match / neg_q if neg_q else np.nan,
                    "candidate_nodes": weighted_avg(group, "avg_candidate_nodes"),
                    "total_s": weighted_avg(group, "avg_total_s"),
                    "solver_ms": weighted_avg(group, "avg_solver_ms"),
                    "false_positives": pd.to_numeric(group["false_positives"], errors="coerce").sum(),
                    "timeouts": pd.to_numeric(group["timeouts"], errors="coerce").sum(),
                }
            )
    return pd.DataFrame(rows)


def savefig(name: str) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PAPER_DIR / name, dpi=240, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close()


def draw_box(ax, xy, wh, text, fc="#f7f7f7", ec="#333333", fontsize=8.5, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        linespacing=1.12,
    )
    return patch


def arrow(ax, start, end, color="#333333", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(10.8, 3.15))
    ax.set_axis_off()
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0.35, 3.55)

    boxes = [
        ((0.20, 2.48), (1.30, 0.58), "Query\nQ", "#e8f1fb"),
        ((0.20, 1.22), (1.30, 0.58), "Data graph\nG", "#f6ece8"),
        ((1.85, 1.22), (1.45, 0.58), "METIS\npartitions", "#fff5d6"),
        ((3.78, 2.48), (1.40, 0.58), "GNN query\nencoder", "#e8f1fb"),
        ((3.78, 1.22), (1.40, 0.58), "Cached partition\nembeddings", "#f0f5e8"),
        ((5.75, 1.86), (1.15, 0.58), "FAISS\nranker", "#f3e8fb"),
        ((7.28, 1.86), (1.22, 0.58), "Overlap\nstitch", "#fff5d6"),
        ((8.88, 1.86), (1.35, 0.58), "Signature +\nlabel pruning", "#f0f5e8"),
        ((8.88, 0.62), (1.35, 0.58), "Glasgow exact\nverifier", "#e8f1fb"),
    ]
    for args in boxes:
        draw_box(ax, *args)

    arrow(ax, (1.50, 2.77), (3.78, 2.77))
    arrow(ax, (1.50, 1.51), (1.85, 1.51))
    arrow(ax, (3.30, 1.51), (3.78, 1.51))
    arrow(ax, (5.18, 2.77), (5.75, 2.25), rad=-0.12)
    arrow(ax, (5.18, 1.51), (5.75, 1.97), rad=0.12)
    arrow(ax, (6.90, 2.15), (7.28, 2.15))
    arrow(ax, (8.50, 2.15), (8.88, 2.15))
    arrow(ax, (9.55, 1.86), (9.55, 1.20))

    draw_box(
        ax,
        (5.20, 3.03),
        (3.35, 0.42),
        "FullCov condition:\ntrue partitions inside retrieved overlap region",
        "#ffffff",
        "#777777",
        fontsize=6.8,
    )
    arrow(ax, (6.88, 3.03), (6.32, 2.45), color="#777777", rad=0.22)
    ax.text(
        9.55,
        0.36,
        "Only solver-certified mappings are accepted.",
        ha="center",
        va="center",
        fontsize=7.4,
    )
    savefig("fig_jigsaw_pipeline.png")


def architecture_figure():
    fig, ax = plt.subplots(figsize=(10.8, 3.95))
    ax.set_axis_off()
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0.45, 3.75)

    draw_box(ax, (0.18, 2.38), (1.18, 0.66), "Node features\n+ edges", "#f6ece8", fontsize=8)
    draw_box(ax, (1.75, 2.38), (1.20, 0.66), "Input projection\n256-d", "#fff5d6", fontsize=8)
    draw_box(
        ax,
        (3.28, 2.18),
        (2.46, 1.06),
        "6 residual message-passing blocks\nGraphSAGE (Cora/Arxiv)\nRGCN (MAG)\nLayerNorm + dropout",
        "#e8f1fb",
        fontsize=7.1,
    )
    draw_box(ax, (5.92, 2.18), (1.60, 1.06), "Multi-layer pooling\nmean / max / sum", "#f0f5e8", fontsize=7.8)
    draw_box(ax, (7.95, 2.38), (1.18, 0.66), "Readout MLP\n+ skip", "#f3e8fb", fontsize=8)
    draw_box(ax, (9.55, 2.38), (1.10, 0.66), "L2 norm\n128-d", "#e8f1fb", fontsize=8)

    for start, end in [
        ((1.36, 2.71), (1.75, 2.71)),
        ((2.95, 2.71), (3.28, 2.71)),
        ((5.74, 2.71), (5.92, 2.71)),
        ((7.52, 2.71), (7.95, 2.71)),
        ((9.13, 2.71), (9.55, 2.71)),
    ]:
        arrow(ax, start, end)

    draw_box(ax, (3.35, 0.82), (2.12, 0.70), "Shared weights embed\nqueries and partitions", "#ffffff", "#777777", fontsize=7.8)
    draw_box(ax, (5.92, 0.82), (1.60, 0.70), "Coarse/fine caches\namortize encoding", "#ffffff", "#777777", fontsize=7.4)
    draw_box(
        ax,
        (7.95, 0.82),
        (2.70, 0.70),
        "Training signal: multi-positive contrastive\n+ hard negatives\n+ weakest-positive FullCov",
        "#ffffff",
        "#777777",
        fontsize=6.9,
    )
    arrow(ax, (4.41, 2.18), (4.41, 1.52), color="#777777")
    arrow(ax, (6.72, 2.18), (6.72, 1.52), color="#777777")
    arrow(ax, (8.54, 2.38), (8.54, 1.52), color="#777777")

    savefig("fig_encoder_architecture.png")


def fullcov_ablation_figure():
    budgets = ["@20", "@50", "@100"]
    control = np.array([21.7, 39.0, 66.0])
    fullcov = np.array([26.0, 48.8, 82.0])
    x = np.arange(len(budgets))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.6, 3.65))
    ax.bar(x - width / 2, control, width, label="Contrastive control", color="#9aa0a6")
    ax.bar(x + width / 2, fullcov, width, label="FullCov objective", color="#1f77b4")
    for xi, c, f in zip(x, control, fullcov):
        ax.text(xi + width / 2, f + 1.5, f"+{f-c:.1f}", ha="center", fontsize=7.6, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"FullCov{b}" for b in budgets])
    ax.set_ylabel("Coverage success (%)")
    ax.set_ylim(0, 92)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    savefig("fig_fullcov_ablation.png")


def design_ablation_figure():
    labels = ["Full\nJigsaw", "No\noverlap", "No\nsignature", "No\ncomponents", "No exact\nlabel"]
    candidate_k = np.array([0.2, 0.2, 46.9, 0.2, 0.4])
    solver_ms = np.array([10.1, 7.4, 10.4, 48.0, 60.6])
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7.3, 3.8))
    bars = ax1.bar(x - 0.17, candidate_k, 0.34, color="#1f77b4", label="Candidate nodes (K)")
    ax1.set_ylabel("Candidate nodes (K)")
    ax1.set_yscale("symlog", linthresh=1)
    ax1.set_ylim(0, 70)
    ax1.grid(axis="y", alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(x + 0.17, solver_ms, marker="o", linewidth=2.0, color="#d62728", label="Solver time (ms)")
    ax2.set_ylabel("Solver time (ms)")
    ax2.set_ylim(0, 70)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    for rect, value in zip(bars, candidate_k):
        ax1.text(rect.get_x() + rect.get_width() / 2, max(value, 0.2) * 1.18, f"{value:.1f}", ha="center", fontsize=7.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    savefig("fig_design_ablation.png")


def production_bar_figure(totals: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.05))
    x = np.arange(len(DATASET_ORDER))
    width = 0.13

    for i, method in enumerate(METHOD_ORDER):
        label = METHOD_LABEL[method]
        sub = totals[totals["method"].eq(method)].set_index("dataset").loc[DATASET_ORDER]
        axes[0].bar(
            x + (i - 2.5) * width,
            sub["positive_solve_rate"],
            width,
            label=label,
            color=METHOD_COLOR[label],
        )
        axes[1].bar(
            x + (i - 2.5) * width,
            sub["candidate_nodes"] / 1000.0,
            width,
            label=label,
            color=METHOD_COLOR[label],
        )
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL[d] for d in DATASET_ORDER])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Positive exact-solve rate", fontsize=9)
    axes[0].set_ylabel("Solved positives (%)")
    axes[0].set_ylim(0, 112)
    axes[1].set_title("Verifier candidate domain", fontsize=9)
    axes[1].set_ylabel("Average candidate nodes (K, log)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.03, 800)
    axes[1].axhline(1, color="#999999", linewidth=0.6, zorder=0)
    axes[0].set_ylabel("Positive exact-solve rate (%)")
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(1.05, 1.22), frameon=False)
    savefig("fig_production_positive_rates.png")


def mag_tradeoff_figure(totals: pd.DataFrame):
    mag = totals[totals["dataset"].eq("mag")].copy()
    mag["cand_k"] = mag["candidate_nodes"] / 1000.0
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    offsets = {
        "Jigsaw": (6, 1.2, "left"),
        "Mean-RRF": (6, -0.2, "left"),
        "MeanFeat": (5, 0.8, "left"),
        "TopoFeat": (5, 0.8, "left"),
        "Random": (5, 0.8, "left"),
        "FilterAll": (5, 0.8, "left"),
    }
    for _, row in mag.iterrows():
        label = row["method_label"]
        ax.scatter(row["cand_k"], row["positive_solve_rate"], s=120, color=METHOD_COLOR[label], edgecolor="black", linewidth=0.8, zorder=3)
        dx, dy, ha = offsets[label]
        ax.text(row["cand_k"] + dx, row["positive_solve_rate"] + dy, label, fontsize=8, ha=ha)
    ax.set_xlabel("Average candidate nodes (thousands)")
    ax.set_ylabel("Positive exact-solve rate (%)")
    ax.set_xlim(180, 490)
    ax.set_ylim(20, 78)
    ax.grid(alpha=0.25)
    savefig("fig_mag_tradeoff.png")


def query_family_heatmap(df: pd.DataFrame):
    order = ["single", "k_hop", "degree_k_hop", "random_walk", "multi_fine", "multi_coarse", "negative_label", "negative_structure"]
    labels = ["Single", "K-hop", "Deg-K", "Walk", "Multi-fine", "Multi-coarse", "Neg-label", "Neg-struct"]
    values = []
    for dataset in DATASET_ORDER:
        row = []
        for qtype in order:
            group = df[(df["dataset"].eq(dataset)) & (df["method"].eq("hybrid")) & (df["query_type"].eq(qtype))]
            pos_q = pd.to_numeric(group["positive_queries"], errors="coerce").sum()
            neg_q = pd.to_numeric(group["negative_queries"], errors="coerce").sum()
            if pos_q:
                row.append(100.0 * pd.to_numeric(group["positive_solved"], errors="coerce").sum() / pos_q)
            else:
                row.append(100.0 * pd.to_numeric(group["correct_no_match"], errors="coerce").sum() / neg_q)
        values.append(row)
    arr = np.array(values)
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticklabels([DATASET_LABEL[d] for d in DATASET_ORDER])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color = "white" if arr[i, j] > 60 else "black"
            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Rate (%)")
    savefig("fig_jigsaw_query_family_heatmap.png")


def partition_overlap_figure():
    # Uses the CORRECTED per-partition boundary-overlap stats (the operator the
    # solver actually sees), NOT the loose "union of whole neighbor partitions"
    # bound. The loose bound is shown only for contrast. See
    # scripts/compute_boundary_overlap_stats.py and runs/diagnostics/boundary_overlap_stats.csv.
    path = DIAGNOSTIC_DIR / "boundary_overlap_stats.csv"
    if not path.exists():
        return
    stats = pd.read_csv(path)
    stats["dataset_label"] = stats["dataset"].map(DATASET_LABEL)
    stats = stats.set_index("dataset").loc[DATASET_ORDER].reset_index()
    x = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    width = 0.26
    ax.bar(x - width, stats["partition_nodes_median"], width,
           label="Coarse partition", color="#1f77b4")
    ax.bar(x, stats["expanded_part_nodes_median"], width,
           label="Partition + 1-hop boundary overlap (actual operator)", color="#2ca02c")
    ax.bar(x + width, stats["full_neighbor_expansion_nodes_median"], width,
           label="Whole-neighbor-partition union (loose bound)", color="#d62728")
    ax.set_yscale("log")
    ax.set_ylabel("Nodes per selected partition (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(stats["dataset_label"])
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    for xi in x:
        mult = stats.loc[xi, "boundary_overlap_multiple"]
        ax.text(xi, stats.loc[xi, "expanded_part_nodes_median"] * 1.25,
                f"{mult:.1f}x", ha="center", va="bottom", fontsize=7.5)
    savefig("fig_partition_overlap_stats.png")


def budget_curves(df: pd.DataFrame):
    budgets = [20, 50, 100, 200, 500, 1000]
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    for dataset in DATASET_ORDER:
        group = df[(df["dataset"].eq(dataset)) & (df["method"].eq("hybrid")) & (pd.to_numeric(df["positive_queries"], errors="coerce") > 0)]
        pos_q = pd.to_numeric(group["positive_queries"], errors="coerce").sum()
        values = [100.0 * pd.to_numeric(group[f"solved_by_{budget}"], errors="coerce").sum() / pos_q for budget in budgets]
        ax.plot(budgets, values, marker="o", linewidth=2.3, label=DATASET_LABEL[dataset])
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(0, 80)
    ax.set_xlabel("Cascade budget")
    ax.set_ylabel("Cumulative positive exact-solve rate (%)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    savefig("fig_jigsaw_budget_curves.png")


def main() -> None:
    df = load_summary_for_figures()
    totals = method_totals(df)
    pipeline_figure()
    architecture_figure()
    fullcov_ablation_figure()
    production_bar_figure(totals)
    mag_tradeoff_figure(totals)
    query_family_heatmap(df)
    partition_overlap_figure()
    budget_curves(df)
    design_ablation_figure()
    print("generated paper figures")


if __name__ == "__main__":
    main()
