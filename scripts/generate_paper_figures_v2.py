"""Regenerate the four stale/broken paper figures from the CANONICAL paper-table values.

Sources (all verified against runs):
  - Production matrix (walk-aware MAG encoder): paper Table production_matrix
    == benchmarks/paper_results + runs/lightning_completion/mag_targeted_v1_final composite.
  - Budget curves: release cumulative solved-by-budget curves quoted in the papers
    (MAG walk-aware; Cora/Arxiv overlap-aware retrained encoders).
  - Family table: paper Table family (walk-aware deployed encoder).

Old figures were stale (generic-mix encoder, pre-walk-aware) and/or had clipped axes
(budget curves ylim 80; MAG tradeoff ylim 78 cut off Jigsaw/Mean-RRF/FilterAll points).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "paper"

METHODS = ["Jigsaw", "Mean-RRF", "MeanFeat", "TopoFeat", "Random", "FilterAll"]
COLORS = {
    "Jigsaw": "#1f77b4",
    "Mean-RRF": "#2ca02c",
    "MeanFeat": "#9467bd",
    "TopoFeat": "#8c564b",
    "Random": "#7f7f7f",
    "FilterAll": "#d62728",
}

# Production matrix: pos-rate (%), avg candidate nodes (K)
POS = {
    "Cora":  {"Jigsaw": 94.2, "Mean-RRF": 93.8, "MeanFeat": 88.8, "TopoFeat": 33.4, "Random": 31.3, "FilterAll": 100.0},
    "Arxiv": {"Jigsaw": 92.8, "Mean-RRF": 94.0, "MeanFeat": 88.9, "TopoFeat": 33.2, "Random": 37.7, "FilterAll": 100.0},
    "MAG":   {"Jigsaw": 88.6, "Mean-RRF": 85.4, "MeanFeat": 64.6, "TopoFeat": 37.1, "Random": 51.4, "FilterAll": 98.4},
}
CAND_K = {
    "Cora":  {"Jigsaw": 2.5, "Mean-RRF": 2.4, "MeanFeat": 3.5, "TopoFeat": 9.5, "Random": 9.6, "FilterAll": 0.8},
    "Arxiv": {"Jigsaw": 0.2, "Mean-RRF": 0.2, "MeanFeat": 0.2, "TopoFeat": 0.3, "Random": 0.3, "FilterAll": 0.1},
    "MAG":   {"Jigsaw": 138.1, "Mean-RRF": 137.1, "MeanFeat": 212.1, "TopoFeat": 235.6, "Random": 282.1, "FilterAll": 443.8},
}

# Cumulative solved-by-budget (paper Sec. results): dataset -> (budgets, values)
BUDGET_CURVES = {
    "Cora":  ([2, 5, 10, 20], [64.0, 82.7, 94.2, 100.0]),
    "Arxiv": ([20, 50, 100, 200], [69.6, 81.3, 92.8, 100.0]),
    "MAG":   ([20, 50, 100, 200, 500, 1000], [37.6, 46.0, 53.0, 61.9, 75.4, 88.6]),
}

# Family table (paper Table family, walk-aware deployed encoder)
FAMILIES = ["Single", "K-hop", "Deg-K", "Walk", "Multi-fine", "Multi-coarse", "Neg-label", "Neg-struct"]
FAMILY_RATES = {
    "Cora":  [100.0, 98.3, 97.3, 75.3, 100.0, 94.3, 100.0, 100.0],
    "Arxiv": [100.0, 97.3, 98.3, 82.3, 100.0, 78.7, 100.0, 100.0],
    "MAG":   [98.7, 93.3, 92.7, 71.0, 100.0, 76.0, 100.0, 99.7],
}


def production_positive_rates():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    datasets = ["Cora", "Arxiv", "MAG"]
    x = np.arange(len(datasets))
    width = 0.13
    for i, m in enumerate(METHODS):
        offs = (i - 2.5) * width
        axes[0].bar(x + offs, [POS[d][m] for d in datasets], width, label=m, color=COLORS[m])
        axes[1].bar(x + offs, [CAND_K[d][m] for d in datasets], width, label=m, color=COLORS[m])
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("Positive exact-solve rate (%)")
    axes[0].set_title("Positive exact-solve rate")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.03, 800)
    axes[1].set_ylabel("Average candidate nodes (K, log)")
    axes[1].set_title("Verifier candidate domain")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig_production_positive_rates.png", dpi=200)
    plt.close(fig)


def budget_curves():
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for name, (budgets, values) in BUDGET_CURVES.items():
        ax.plot(budgets, values, marker="o", linewidth=2.3, label=name)
    ax.set_xscale("log")
    ticks = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Cascade budget (coarse partitions, log)")
    ax.set_ylabel("Cumulative positive exact-solve rate (%)")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_jigsaw_budget_curves.png", dpi=200)
    plt.close(fig)


def family_heatmap():
    data = np.array([FAMILY_RATES["Cora"], FAMILY_RATES["Arxiv"], FAMILY_RATES["MAG"]])
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels(FAMILIES, rotation=20)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Cora (K=10)", "Arxiv (K=100)", "MAG (K=1000)"])
    for i in range(3):
        for j in range(len(FAMILIES)):
            v = data[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v > 55 else "black", fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rate (%)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_jigsaw_query_family_heatmap.png", dpi=200)
    plt.close(fig)


def mag_tradeoff():
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    for m in METHODS:
        x = CAND_K["MAG"][m]
        y = POS["MAG"][m]
        ax.scatter([x], [y], s=220, color=COLORS[m], edgecolor="black", zorder=3)
        dx, dy = 10, 0
        if m == "Mean-RRF":
            dy = -4.5
        ax.annotate(m, (x, y), xytext=(x + dx, y + dy), fontsize=12)
    ax.set_xlim(120, 480)
    ax.set_ylim(25, 105)
    ax.set_xlabel("Average candidate nodes (thousands)")
    ax.set_ylabel("Positive exact-solve rate (%)")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mag_tradeoff.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    production_positive_rates()
    budget_curves()
    family_heatmap()
    mag_tradeoff()
    print("wrote fig_production_positive_rates / fig_jigsaw_budget_curves / fig_jigsaw_query_family_heatmap / fig_mag_tradeoff")
