#!/usr/bin/env python3
"""Regenerate the headline paper numbers from local run artifacts.

Required checks fail if a load-bearing paper number no longer matches the local
CSV evidence. Optional external-diagnostic provenance is reported separately so
missing non-core logs do not mask the status of the main Jigsaw tables.

Covered:
  * MAG walk-aware production matrix   (Jigsaw 88.6%, Mean-RRF 85.4%)
  * MAG per-family solve rates          (98.7/93.3/92.7/71.0/100/76.0)
  * MAG per-size slices @K=1000         (93.2/90.7/82.0)
  * Jigsaw-vs-Mean-RRF paired McNemar   (92/35, p=4.36e-7)
  * Cross-dataset FeatureIndex/learned selector (Cora/Arxiv, n=540 each)
  * Inference-time retrieval-remedy foreclosure (MAG FullCov@1000)
  * Optional GNN-PE public-release diagnostic provenance

Run from the repo root:  python scripts/analysis/reproduce_paper_numbers.py
Use --strict-optional to make missing optional diagnostic provenance fail.
"""
import csv, glob, os, sys, statistics
from math import comb

CAP = 1000  # MAG fair budget = 1000 partitions (50% of 2,000)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAMILIES = ["single", "k_hop", "degree_k_hop", "multi_fine", "multi_coarse", "random_walk"]
SPATIAL_FAMILIES = ["degree_k_hop", "k_hop", "random_walk"]


def _T(v):
    return str(v).strip().lower() == "true"


def _load(f):
    # Windows caps paths at 260 chars; the long per-query filenames need the
    # extended-length prefix. No-op on POSIX (where reviewers will run this).
    if os.name == "nt":
        fabs = os.path.abspath(f)
        if len(fabs) > 240 and not fabs.startswith("\\\\?\\"):
            f = "\\\\?\\" + fabs
    with open(f, newline="") as fh:
        return list(csv.DictReader(fh))


def _g(pat):
    return [x for x in glob.glob(os.path.join(ROOT, pat)) if "partial" not in x]


def _load_rel(rel):
    return _load(os.path.join(ROOT, rel))


def _seedtag(f):
    return "A" if "20260607" in f else "B"


def _solved_map(files, method, model=None, cap=CAP):
    """query key -> solved within budget<=cap (any solver_found True)."""
    m = {}
    for f in files:
        for r in _load(f):
            if r.get("method") != method:
                continue
            if model and r.get("model") not in (model,):
                continue
            try:
                b = float(r["budget"])
            except (KeyError, ValueError):
                continue
            if b > cap:
                continue
            k = (r["query_type"], r["target_query_size"], r["query_id"], _seedtag(f))
            m[k] = m.get(k, False) or _T(r.get("solver_found", ""))
    return m


def _pct(d, keys=None):
    keys = keys or list(d)
    return 100.0 * sum(d[k] for k in keys) / max(1, len(keys))


def _pct_rows(rows, col):
    return 100.0 * sum(int(float(r[col])) for r in rows) / max(1, len(rows))


def _median_col(rows, col):
    return float(statistics.median(int(float(r[col])) for r in rows))


def _mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n))


def check(label, got, want, tol=0.15):
    ok = abs(got - want) <= tol
    print(f"  [{'OK ' if ok else 'XX '}] {label:44s} computed={got:7.2f}  paper={want:7.2f}")
    return ok


def check_count(label, got, want):
    ok = got == want
    print(f"  [{'OK ' if ok else 'XX '}] {label:44s} computed={got:7d}  paper={want:7d}")
    return ok


def check_less(label, got, threshold):
    ok = got < threshold
    print(f"  [{'OK ' if ok else 'XX '}] {label:44s} computed={got:.2e}  required<{threshold:.0e}")
    return ok


def check_warn(label, message):
    print(f"  [WARN] {label:44s} {message}")


def _selector_checks():
    ok = True
    print("=" * 78)
    print("CROSS-DATASET LABEL SELECTIVITY SELECTOR (retrieval rank, n=540/dataset)")
    print("=" * 78)

    datasets = {
        "Cora": {
            "file": "runs/fi_selectivity/cora_fi_neural.csv",
            "medians": {"max_true_rank_fi_feature": 2.0, "max_true_rank_fi_class": 7.0, "max_true_rank_neural": 5.0},
            "feature_counts": (352, 6, 182),  # FI better, neural better, tie
            "class_counts": (358, 111, 71),   # neural better, FI better, tie
        },
        "Arxiv": {
            "file": "runs/fi_selectivity/arxiv_fi.csv",
            "medians": {"max_true_rank_fi_feature": 4.0, "max_true_rank_fi_class": 104.0, "max_true_rank_neural": 44.0},
            "feature_counts": (388, 1, 151),
            "class_counts": (452, 76, 12),
        },
    }

    for name, cfg in datasets.items():
        rows = _load_rel(cfg["file"])
        print(f"{name}:")
        ok &= check_count("queries", len(rows), 540)
        for col, want in cfg["medians"].items():
            # Arxiv FI-class has an even-n median of 104.5 in the raw zero-based
            # ranks; the table reports the rounded headline value 104.
            ok &= check(f"median {col}", _median_col(rows, col), want, tol=0.55)

        ff = sum(int(r["max_true_rank_fi_feature"]) < int(r["max_true_rank_neural"]) for r in rows)
        nf = sum(int(r["max_true_rank_neural"]) < int(r["max_true_rank_fi_feature"]) for r in rows)
        tf = len(rows) - ff - nf
        ok &= check_count("feature-label FI wins", ff, cfg["feature_counts"][0])
        ok &= check_count("feature-label neural wins", nf, cfg["feature_counts"][1])
        ok &= check_count("feature-label ties", tf, cfg["feature_counts"][2])
        ok &= check_less("feature-label sign-test p", _mcnemar_exact_p(ff, nf), 1e-30)

        cn = sum(int(r["max_true_rank_neural"]) < int(r["max_true_rank_fi_class"]) for r in rows)
        fc = sum(int(r["max_true_rank_fi_class"]) < int(r["max_true_rank_neural"]) for r in rows)
        tc = len(rows) - cn - fc
        ok &= check_count("class-label neural wins", cn, cfg["class_counts"][0])
        ok &= check_count("class-label FI wins", fc, cfg["class_counts"][1])
        ok &= check_count("class-label ties", tc, cfg["class_counts"][2])
        ok &= check_less("class-label sign-test p", _mcnemar_exact_p(cn, fc), 1e-30)
    return ok


def _foreclosure_checks():
    ok = True
    print("=" * 78)
    print("INFERENCE-TIME RETRIEVAL REMEDY FORECLOSURE (MAG FullCov@1000)")
    print("=" * 78)

    fine = _load_rel("runs/probe_finegrain/probe_finegrain_per_query.csv")
    multi = _load_rel("runs/multivector_probe/probe_multivector_per_query.csv")
    expected_fine = {
        "degree_k_hop": {"single": 17.3, "fine_parent": 16.7, "fine_overlap": 17.3, "fine_overlap2": 18.3, "subq_fine": 16.0, "diff1": 19.7, "stitch": 4.0},
        "k_hop": {"single": 14.7, "fine_parent": 12.7, "fine_overlap": 13.3, "fine_overlap2": 13.3, "subq_fine": 12.7, "diff1": 17.0, "stitch": 3.0},
        "random_walk": {"single": 9.0, "fine_parent": 8.3, "fine_overlap": 10.0, "fine_overlap2": 9.3, "subq_fine": 8.7, "diff1": 10.3, "stitch": 0.7},
    }
    expected_multi = {
        "degree_k_hop": {"subq8_max": 17.7, "subq8_top2": 17.7, "subq4_max": 17.7},
        "k_hop": {"subq8_max": 14.3, "subq8_top2": 14.7, "subq4_max": 14.3},
        "random_walk": {"subq8_max": 8.0, "subq8_top2": 8.7, "subq4_max": 7.7},
    }

    max_gain = -999.0
    for fam in SPATIAL_FAMILIES:
        fine_rows = [r for r in fine if r["query_type"] == fam]
        multi_rows = [r for r in multi if r["query_type"] == fam]
        ok &= check_count(f"{fam} fine rows", len(fine_rows), 300)
        ok &= check_count(f"{fam} multivector rows", len(multi_rows), 300)
        baseline = expected_fine[fam]["single"]
        for method, want in expected_fine[fam].items():
            got = _pct_rows(fine_rows, f"fullcov1000_{method}")
            ok &= check(f"{fam} {method}", got, want)
            max_gain = max(max_gain, got - baseline)
        for method, want in expected_multi[fam].items():
            got = _pct_rows(multi_rows, f"fullcov1000_{method}")
            ok &= check(f"{fam} {method}", got, want)
            max_gain = max(max_gain, got - baseline)
    ok &= check("best gain over single-vector baseline", max_gain, 2.3, tol=0.15)
    ok &= check_less("genuine-fix gate", max_gain, 20.0)
    return ok


def _gnnpe_optional_checks(strict=False):
    ok = True
    print("=" * 78)
    print("OPTIONAL GNN-PE PUBLIC-RELEASE DIAGNOSTIC PROVENANCE")
    print("=" * 78)

    v38 = _g("runs/gnnpe_spike/*v38*e4*/answer_summary.csv")
    if v38:
        rows = _load(v38[0])
        found = sum(int(float(r.get("answer") or 0)) > 0 for r in rows)
        size100 = [r for r in rows if str(r.get("target_size")) == "100"]
        found100 = sum(int(float(r.get("answer") or 0)) > 0 for r in size100)
        ok &= check_count("GNN-PE e=4 recovered", found, 7)
        ok &= check_count("GNN-PE e=4 queries", len(rows), 24)
        ok &= check_count("GNN-PE e=4 recovered @n=100", found100, 0)
    else:
        check_warn("GNN-PE e=4", "answer_summary.csv not found")
        ok &= not strict

    v59 = _g("runs/gnnpe_spike/*v59*e128*/answer_summary.csv")
    if v59:
        rows = _load(v59[0])
        found = sum(int(float(r.get("answer") or 0)) > 0 for r in rows)
        size100 = [r for r in rows if str(r.get("target_size")) == "100"]
        found100 = sum(int(float(r.get("answer") or 0)) > 0 for r in size100)
        ok &= check_count("GNN-PE e=128 recovered", found, 5)
        ok &= check_count("GNN-PE e=128 queries", len(rows), 24)
        ok &= check_count("GNN-PE e=128 recovered @n=100", found100, 0)
    else:
        check_warn("GNN-PE e=128", "compact answer_summary.csv not found locally; launcher/query provenance remains")
        ok &= not strict
    return ok


def main():
    strict_optional = "--strict-optional" in sys.argv[1:]
    ok = True
    print("=" * 78)
    print("MAG WALK-AWARE (deployed encoder), fair budget K<=1000, 2 seeds, n=300/family")
    print("=" * 78)

    jig_files = (_g("runs/lightning_completion/mag_walkaware_remaining_v2/final_per_query/*.csv")
                 + _g("runs/lightning_completion/mag_targeted_v1_final/results/*per_query.csv"))
    J = _solved_map(jig_files, "hybrid", model="mag_walkaware_best")

    paper_fam = {"single": 98.7, "k_hop": 93.3, "degree_k_hop": 92.7,
                 "multi_fine": 100.0, "multi_coarse": 76.0, "random_walk": 71.0}
    print("Jigsaw per-family solve%:")
    for fam in FAMILIES:
        keys = [k for k in J if k[0] == fam]
        ok &= check(fam, _pct(J, keys), paper_fam[fam])
    ok &= check("OVERALL (Jigsaw)", _pct(J), 88.6)

    print("Jigsaw per-size slice @K=1000:")
    paper_size = {"20": 93.2, "50": 90.7, "100": 82.0}
    for sz in ("20", "50", "100"):
        keys = [k for k in J if k[1] == sz]
        ok &= check(f"size {sz}", _pct(J, keys), paper_size[sz])

    # Mean-RRF (walk-aware): easy families rerun + hard families from targeted run
    mrrf_files = (_g("runs/lightning_completion/mag_walkaware_mrrf_easy_v1/final_per_query/*mean_rrf*per_query.csv")
                  + _g("runs/lightning_completion/mag_targeted_v1_final/results/*mean_rrf*per_query.csv"))
    R = _solved_map(mrrf_files, "coarse_mean_rrf", model="mag_walkaware_best")
    ok &= check("OVERALL (Mean-RRF)", _pct(R), 85.4)

    print("Paired McNemar Jigsaw vs Mean-RRF (matched queries):")
    common = sorted(set(J) & set(R))
    b = sum(1 for k in common if J[k] and not R[k])
    c = sum(1 for k in common if R[k] and not J[k])
    p = _mcnemar_exact_p(b, c)
    print(f"  n_matched={len(common)}  Jigsaw-only(b)={b}  Mean-RRF-only(c)={c}  exact p={p:.2e}")
    ok &= check("McNemar b (Jigsaw wins)", b, 92, tol=3)
    ok &= check("McNemar c (Mean-RRF wins)", c, 35, tol=3)

    ok &= _selector_checks()
    ok &= _foreclosure_checks()
    optional_ok = _gnnpe_optional_checks(strict=strict_optional)
    ok &= optional_ok

    print("=" * 78)
    print("ALL REQUIRED CHECKS PASSED" if ok else "SOME CHECKS FAILED (see XX/WARN above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
