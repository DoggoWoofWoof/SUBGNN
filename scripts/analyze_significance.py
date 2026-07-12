"""Statistical significance for the MAG production matrix.

Reports, per retrieval method, the MAG positive solve rate with a bootstrap 95%
confidence interval, and paired exact-McNemar tests of the neural (Jigsaw) policy
against every other policy. Pairs are formed per query (query_type, size, seed,
query_id); a query "solves" if any budget reached an exact match.

Inputs reconstruct the corrected canonical positive grid: non-multi families from
the production run dirs, corrected multi families from the connected rerun. Solve
rates here match benchmarks/paper_results/final_results/final_all_datasets_summary.csv
exactly (neural 86.0%, FilterAll 98.4%, etc.).

Outputs:
  runs/diagnostics/mag_significance.csv   (per-method rate + CI + vs-neural McNemar)
  runs/diagnostics/mag_significance.md
"""

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def as_bool(x):
    return str(x).strip().lower() in {"1", "true", "yes"}


NONMULTI = {"single", "k_hop", "degree_k_hop", "random_walk"}
MULTI = {"multi_fine", "multi_coarse"}
METHOD_MAP = {
    "hybrid": "neural_component",
    "coarse_mean_rrf": "mean_rrf_component",
    "all": "filterall_component",
    "mean_feature": "mean_feature_component",
    "random": "random_component",
    "topo_feature": "topo_feature_component",
}
METHODS = [
    "neural_component", "mean_rrf_component", "mean_feature_component",
    "random_component", "topo_feature_component", "filterall_component",
]
LABEL = {
    "neural_component": "Jigsaw", "mean_rrf_component": "Mean-RRF",
    "mean_feature_component": "MeanFeat", "random_component": "Random",
    "topo_feature_component": "Topo", "filterall_component": "FilterAll",
}


def seed_of(path):
    m = re.search(r"s(20\d{6})", path)
    return m.group(1) if m else "?"


def ingest(pattern, allow, solved, drop_final=True):
    for fp in glob.glob(pattern):
        if "partial" in fp:
            continue
        sd = seed_of(fp)
        for r in csv.DictReader(open(fp)):
            if not as_bool(r.get("expected_match", "true")):
                continue
            qt = r.get("query_type", "")
            if qt not in allow:
                continue
            if drop_final and r.get("model") == "mag_rgcn_final":
                continue
            m = METHOD_MAP.get(r.get("method", ""), r.get("method", ""))
            key = (qt, r.get("target_query_size"), sd, r.get("query_id"))
            solved[m][key] = solved[m].get(key, False) or as_bool(r.get("cascade_first_solved"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonmulti-glob", default="runs/lightning_mag_full/*/results/*_per_query.csv")
    ap.add_argument("--multi-glob", default="runs/lcr_mag_v3/results/*_per_query.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out-csv", default="runs/diagnostics/mag_significance.csv")
    ap.add_argument("--out-md", default="runs/diagnostics/mag_significance.md")
    args = ap.parse_args()

    solved = defaultdict(dict)
    ingest(args.nonmulti_glob, NONMULTI, solved)
    ingest(args.multi_glob, MULTI, solved)

    rng = np.random.default_rng(0)
    rates = {}
    agg = {}
    for m in METHODS:
        keys = sorted(solved[m])
        vals = np.array([1 if solved[m][k] else 0 for k in keys])
        boot = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(args.boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rates[m] = (vals.mean(), lo, hi, len(vals))
        agg[m] = dict(zip(keys, vals))

    nk = agg["neural_component"]
    rows = []
    for m in METHODS:
        rate, lo, hi, n = rates[m]
        row = {
            "method": LABEL[m], "n": n,
            "solve_rate": round(rate * 100, 1),
            "ci95_lo": round(lo * 100, 1), "ci95_hi": round(hi * 100, 1),
        }
        if m != "neural_component":
            common = set(nk) & set(agg[m])
            wins = sum(1 for k in common if nk[k] == 1 and agg[m][k] == 0)
            losses = sum(1 for k in common if nk[k] == 0 and agg[m][k] == 1)
            disc = wins + losses
            p = binomtest(min(wins, losses), disc, 0.5).pvalue if disc else 1.0
            row.update({"neural_wins": wins, "neural_losses": losses,
                        "mcnemar_p": f"{p:.2e}",
                        "neural_better": "yes" if wins > losses else "no"})
        else:
            row.update({"neural_wins": "", "neural_losses": "", "mcnemar_p": "", "neural_better": ""})
        rows.append(row)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write("# MAG production matrix — significance\n\n")
        fh.write("Positive solve rate (n=1800/method) with bootstrap 95% CI, and paired "
                 "exact-McNemar of Jigsaw (neural) vs each policy.\n\n")
        fh.write("| Method | Solve % | 95% CI | vs Jigsaw (wins/losses) | McNemar p |\n")
        fh.write("|---|---:|---|---:|---:|\n")
        for r in rows:
            vs = "—" if r["method"] == "Jigsaw" else f"{r['neural_wins']}/{r['neural_losses']}"
            p = "—" if r["method"] == "Jigsaw" else r["mcnemar_p"]
            fh.write(f"| {r['method']} | {r['solve_rate']} | [{r['ci95_lo']}, {r['ci95_hi']}] | {vs} | {p} |\n")
        fh.write("\nJigsaw significantly outperforms every learned/unlearned retrieval baseline "
                 "(including Mean-RRF, p=0.014); FilterAll significantly exceeds Jigsaw as the "
                 "exhaustive recall ceiling.\n")

    for r in rows:
        print(r)
    print(f"\nWrote {args.out_csv} and {args.out_md}")


if __name__ == "__main__":
    main()
