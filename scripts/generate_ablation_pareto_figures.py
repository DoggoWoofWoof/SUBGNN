"""Generate the combined MAG+Arxiv design-ablation figure and the memory<->latency
Pareto figure from the final verified runs (2026-07). Values hardcoded from:
  - runs/mag_design_ablation_v2_dl (48-query shared set)
  - runs/arxiv_design_ablation_v1_dl (48-query shared set)
  - runs/{cora_arxiv_pareto_v1_dl, mag_pareto_v1_dl} (cache sweep)
Writes paper/fig_design_ablation.png and paper/fig_memory_latency.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "paper"

VARIANTS = ["Full", "No\noverlap", "No\nsignature", "No\ncomponents", "No\nexact-label"]
MAG_SOLVE = [88.9, 44.4, 88.9, 63.9, 52.8]           # positive solve rate (positives only, /36)
ARXIV_CAND = [94, 97, 7081, 94, 223]                 # avg candidate nodes


def design_ablation():
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(VARIANTS))
    # MAG: solve rate; highlight the overlap collapse
    colors = ["#1f77b4", "#d62728", "#1f77b4", "#ff7f0e", "#ff7f0e"]
    ax[0].bar(x, MAG_SOLVE, color=colors)
    ax[0].axhline(88.9, ls="--", lw=1, color="#555")
    ax[0].set_title("MAG: overlap is the lever\n(positive solve rate, shared query set)")
    ax[0].set_ylabel("Positive solve rate (%)")
    ax[0].set_ylim(0, 105)
    for i, v in enumerate(MAG_SOLVE):
        ax[0].text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
    # Arxiv: candidate nodes (log); highlight the signature explosion
    colorsa = ["#1f77b4", "#1f77b4", "#d62728", "#1f77b4", "#ff7f0e"]
    ax[1].bar(x, ARXIV_CAND, color=colorsa)
    ax[1].set_yscale("log")
    ax[1].set_title("Arxiv: signature is the lever\n(candidate nodes, log)")
    ax[1].set_ylabel("Avg candidate nodes")
    ax[1].set_ylim(50, 20000)
    for i, v in enumerate(ARXIV_CAND):
        ax[1].text(i, v * 1.15, f"{v:,}", ha="center", fontsize=9)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(VARIANTS, fontsize=8.5)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_design_ablation.png", dpi=200)
    plt.close(fig)


# Pareto: peak RSS (GB) vs latency; streaming cache sweep + resident anchor
# MAG cache sweep (GB, mean s, p50 s)
MAG_PTS = [(1.884, 17.89, 5.91), (1.982, 17.83, 5.48), (2.095, 17.82, 5.74),
           (2.293, 17.76, 5.69), (2.604, 17.62, 5.27), (3.045, 17.63, 4.83)]
MAG_RESIDENT_GB = 10.2  # canonical resident baseline (mag_baseline_rss.csv 10198 MB); matches body text
MAG_INMEM_S = 6.97
MAG_CLASSICAL_COLD_S = 8.4


def memory_latency():
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.2))
    ds = ["Cora", "Arxiv", "MAG"]
    xx = np.arange(3)
    w = 0.36
    # PANEL 1: memory reduction bars (streaming vs whole-graph resident)
    stream = [0.797, 0.628, 1.884]
    resident = [1.185, 0.958, 10.2]
    ax[0].bar(xx - w / 2, stream, w, label="Jigsaw streaming", color="#1f77b4")
    ax[0].bar(xx + w / 2, resident, w, label="Whole-graph resident", color="#d62728")
    ax[0].set_yscale("log"); ax[0].set_ylim(0.4, 30)
    ax[0].set_ylabel("Peak RSS (GB, log)")
    ax[0].set_title("(a) Memory: bounded at any scale", fontsize=11.5)
    ax[0].set_xticks(xx); ax[0].set_xticklabels(ds)
    for i, (s, r) in enumerate(zip(stream, resident)):
        ax[0].text(i - w / 2, s * 1.05, f"{s:.1f}", ha="center", va="bottom", fontsize=9)
        ax[0].text(i + w / 2, r * 1.05, f"{r:.1f}", ha="center", va="bottom", fontsize=9)
        ax[0].text(i, r * 1.35, f"{r/s:.1f}×", ha="center", va="bottom", fontsize=10.5,
                   color="#1f77b4", fontweight="bold")
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    ax[0].spines[["top", "right"]].set_visible(False); ax[0].grid(axis="y", alpha=0.3)
    # PANEL 2: in-memory latency -- Jigsaw resident vs classical resident (same paired run)
    jig = [0.078, 0.233, 6.97]        # Arxiv 0.233 = neural cascade (the "Jigsaw" row); mean_rrf is 0.217
    clf = [0.183, 0.70, 8.4]          # Cora 0.183 (canonical) -> 2.35x, matches paper text
    mult = ["2.35×", "3.0×", "1.2×"]  # neural cascade: 0.70/0.233 = 3.0x on Arxiv
    ax[1].bar(xx - w / 2, jig, w, label="Jigsaw (in-memory)", color="#1f77b4")
    ax[1].bar(xx + w / 2, clf, w, label="Classical CFL/DP-iso/GQL$^\\dagger$", color="#d62728")
    ax[1].set_yscale("log"); ax[1].set_ylim(0.04, 60)
    ax[1].set_ylabel("Latency per query (s, log)")
    ax[1].set_title("(b) In-memory latency (both resident)", fontsize=11.5)
    ax[1].set_xticks(xx); ax[1].set_xticklabels(ds)
    _fmt = lambda v: f"{v:.3f}" if v < 1 else f"{v:.1f}"
    for i, (j, c) in enumerate(zip(jig, clf)):
        ax[1].text(i - w / 2, j * 1.13, _fmt(j), ha="center", va="bottom", fontsize=9)
        ax[1].text(i + w / 2, c * 1.13, _fmt(c), ha="center", va="bottom", fontsize=9)
        ax[1].text(i, max(j, c) * 1.9, mult[i], ha="center", va="bottom", fontsize=10.5,
                   color="#1f77b4", fontweight="bold")
    ax[1].legend(frameon=False, fontsize=9, loc="upper left")
    ax[1].spines[["top", "right"]].set_visible(False); ax[1].grid(axis="y", alpha=0.3)
    # PANEL 3: MAG memory<->latency Pareto frontier (cache sweep)
    rss = [p[0] for p in MAG_PTS]; mean = [p[1] for p in MAG_PTS]; p50 = [p[2] for p in MAG_PTS]
    ax[2].axhspan(0, MAG_CLASSICAL_COLD_S, color="#2ca02c", alpha=0.06)
    ax[2].plot(rss, mean, "o-", color="#1f77b4", lw=2, ms=6, label="streaming mean")
    ax[2].plot(rss, p50, "s-", color="#2ca02c", lw=2, ms=6, label="streaming p50")
    ax[2].axhline(MAG_CLASSICAL_COLD_S, ls=":", color="#d62728", lw=1.4, label="classical cold-load 8.4 s")
    ax[2].axhline(MAG_INMEM_S, ls="--", color="#7f7f7f", lw=1.2, label="in-mem cascade 6.97 s")
    ax[2].annotate("cache 8", (rss[0], p50[0]), xytext=(rss[0] + 0.05, p50[0] + 1.0), fontsize=8)
    ax[2].annotate("cache 256", (rss[-1], p50[-1]), xytext=(rss[-1] - 0.42, p50[-1] - 1.4), fontsize=8)
    ax[2].set_xlabel("Peak RSS (GB)")
    ax[2].set_ylabel("Latency per query (s)")
    ax[2].set_title("(c) MAG Pareto: more cache buys p50, not mean", fontsize=11.5)
    ax[2].set_xlim(1.6, 3.3); ax[2].set_ylim(0, 20)
    ax[2].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 0.60), ncol=1)
    ax[2].spines[["top", "right"]].set_visible(False); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_memory_latency.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    design_ablation()
    memory_latency()
    print("wrote fig_design_ablation.png (MAG+Arxiv) and fig_memory_latency.png")
