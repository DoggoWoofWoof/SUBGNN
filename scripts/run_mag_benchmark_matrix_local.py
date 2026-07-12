"""Run the MAG production cascade matrix inside one local/Lightning job.

This is the non-Modal port of `run_overlap_cascade_batch`: one job, internal
parallel workers, cached query generation, and result upload handled by the
caller shell script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


def clean_tag(text: str) -> str:
    return str(text).replace(",", "_").replace(" ", "").replace("/", "_")


def method_config(name: str, signature: str, budgets: str, full_budget: str):
    configs = {
        "fullgraph": ("all", "none", full_budget, False, False, False, False),
        "filterall_component": ("all", signature, full_budget, True, True, True, False),
        "neural_component": ("hybrid", signature, budgets, True, True, True, True),
        "random_component": ("random", signature, budgets, True, True, True, False),
        "mean_feature_component": ("mean_feature", signature, budgets, True, True, True, False),
        "mean_rrf_component": ("coarse_mean_rrf", signature, budgets, True, True, True, True),
        "topo_feature_component": ("topo_feature", signature, budgets, True, True, True, False),
        # Classical filter-and-verify structural feature index baseline (GraphGrep/gIndex style).
        "feature_index_component": ("feature_index", signature, budgets, True, True, True, False),
        "neural_no_component": ("hybrid", signature, budgets, False, True, True, True),
        "neural_no_overlap": ("hybrid", signature, budgets, True, True, False, True),
        "neural_no_signature": ("hybrid", "none", budgets, True, True, True, True),
        "neural_no_exact_label": ("hybrid", signature, budgets, True, False, True, True),
        # Pareto: neural with selective overlap (same as neural_component except the
        # overlap operator). Extra overlap flags injected via OVERLAP_POLICY_FLAGS.
        "neural_selective": ("hybrid", signature, budgets, True, True, True, True),
        "neural_selective_topk": ("hybrid", signature, budgets, True, True, True, True),
        # Classical external baseline: full-graph exact Glasgow, no retrieval, no
        # overlap, no signature/label pruning, no component split -> fullgraph fast path.
        "filterall_raw": ("all", "none", full_budget, False, False, False, False),
    }
    if name not in configs:
        raise ValueError(f"Unknown method {name}; choices={sorted(configs)}")
    return configs[name]


def manifest_has_query_cache(cache_dir: str, seed: int, queries: int, target_sizes: str, query_types: str) -> bool:
    root = Path(cache_dir)
    manifest_path = root / "query_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    spec_key = f"{seed}|{queries}|{target_sizes}|{query_types}"
    filename = manifest.get(spec_key) or manifest.get(str(seed))
    if not filename or not (root / filename).exists():
        return False
    try:
        import benchmark_overlap_glasgow_cascade as cascade

        cached = cascade.torch_load_any(root / filename)
        sizes = cascade.parse_budgets(target_sizes)
        return cascade._queries_match_spec(cached, queries, sizes, query_types)
    except Exception as exc:
        print(f"[QUERY CACHE] invalid cached query manifest seed={seed}: {exc}; regenerating", flush=True)
        return False


# Selective-overlap operator flags injected per method (Pareto experiment).
OVERLAP_POLICY_FLAGS = {
    "neural_selective": ["--overlap-max-parts", "8", "--overlap-label-compatible"],
    "neural_selective_topk": ["--overlap-max-parts", "8"],
}


def run_command(tag: str, cmd: list[str], log_path: Path, cwd: Path) -> dict[str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"Running: {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{tag}] {line}", end="")
            log.write(line)
        proc.wait()
    status = "ok" if proc.returncode == 0 else f"failed:{proc.returncode}"
    return {"tag": tag, "status": status, "log": str(log_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mag")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--target-sizes", default="20,50,100")
    parser.add_argument("--query-types", default="all")
    parser.add_argument("--seeds", default="20260607,20260608")
    parser.add_argument("--methods", default="neural_component,random_component,mean_feature_component,mean_rrf_component,filterall_component")
    parser.add_argument("--ablation-set", choices=["none", "full"], default="none")
    parser.add_argument("--model-specs", default="", help="semicolon-separated label=path for encoder methods")
    parser.add_argument(
        "--hierarchy-path",
        default="",
        help="Optional hierarchy .pt path. When omitted, benchmark_overlap_glasgow_cascade.py uses/builds the dataset default.",
    )
    parser.add_argument("--budgets", default="20,50,100,200,500,1000")
    parser.add_argument("--full-budget", default="2000")
    parser.add_argument("--signature", default="type_rel_feat32")
    parser.add_argument("--solver-timeout", type=float, default=5.0)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-dir", default="cache/overlap_cascade")
    parser.add_argument("--output-dir", default="runs/lightning_mag_benchmark_results")
    parser.add_argument("--output-prefix", default="prod_mag_rgcn")
    parser.add_argument("--glasgow-bin", default=os.environ.get("GLASGOW_SOLVER_BIN", "glasgow_subgraph_solver"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--parallel-mode", choices=["task", "query"], default="task")
    parser.add_argument("--max-component-diag-nodes", type=int, default=50000)
    parser.add_argument("--max-component-solver-components", type=int, default=50)
    parser.add_argument("--max-eval-queries", type=int, default=0)
    parser.add_argument("--label-source", default="feature",
                        help="node label for matching/pruning/index: 'feature' (per-feature-vector md5 hash), "
                             "'class' (real class label data.y), or 'feature_bucket_K' (md5(feature) mod K, for a "
                             "selectivity sweep). Passed through to benchmark_overlap_glasgow_cascade.py.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    out_dir = Path(args.output_dir)
    result_dir = out_dir / "results"
    log_dir = out_dir / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    if args.hierarchy_path and not Path(args.hierarchy_path).exists():
        raise FileNotFoundError(f"Hierarchy not found: {args.hierarchy_path}")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if methods and not shutil_which(args.glasgow_bin):
        raise FileNotFoundError(f"Glasgow solver not found: {args.glasgow_bin}")

    if args.ablation_set == "full":
        for m in ["neural_no_component", "neural_no_overlap", "neural_no_signature", "neural_no_exact_label"]:
            if m not in methods:
                methods.append(m)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    py = sys.executable

    # Build query cache once per seed. This makes all methods use the same queries.
    for seed in seeds:
        if manifest_has_query_cache(args.cache_dir, seed, args.queries, args.target_sizes, args.query_types):
            print(f"[QUERY CACHE] manifest hit seed={seed}; skipping generation", flush=True)
            continue
        tag = (
            f"{args.output_prefix}_s{seed}_q{args.queries}_types_{clean_tag(args.query_types)}"
            f"_sizes{clean_tag(args.target_sizes)}_query_cache"
        )
        cmd = [
            py, "scripts/benchmark_overlap_glasgow_cascade.py",
            "--dataset", args.dataset,
            "--queries", str(args.queries),
            "--target-sizes", args.target_sizes,
            "--query-types", args.query_types,
            "--seed", str(seed),
            "--data-root", args.data_root,
            "--hierarchy-path", args.hierarchy_path,
            "--output-prefix", str(result_dir / tag),
            "--budgets", args.budgets,
            "--method", "random",
            "--signature", args.signature,
            "--solver-timeout", str(args.solver_timeout),
            "--glasgow-bin", args.glasgow_bin,
            "--cache-dir", args.cache_dir,
            "--generate-query-cache-only",
        ]
        run_command(tag, cmd, log_dir / f"{tag}.log", root)

    tasks = []
    for seed in seeds:
        for method in methods:
            cascade_method, sig, method_budgets, component_solve, prune_labels, use_overlap, needs_models = method_config(
                method, args.signature, args.budgets, args.full_budget
            )
            tag = (
                f"{args.output_prefix}_s{seed}_q{args.queries}_types_{clean_tag(args.query_types)}"
                f"_sizes{clean_tag(args.target_sizes)}_{method}_b{clean_tag(method_budgets)}"
            )
            output_prefix = result_dir / tag
            summary = Path(f"{output_prefix}_summary.csv")
            per_query = Path(f"{output_prefix}_per_query.csv")
            if args.skip_existing and summary.exists() and per_query.exists():
                print(f"[SKIP] {tag}", flush=True)
                continue
            cmd = [
                py, "scripts/benchmark_overlap_glasgow_cascade.py",
                "--dataset", args.dataset,
                "--queries", str(args.queries),
                "--target-sizes", args.target_sizes,
                "--query-types", args.query_types,
                "--seed", str(seed),
                "--data-root", args.data_root,
                "--hierarchy-path", args.hierarchy_path,
                "--output-prefix", str(output_prefix),
                "--budgets", method_budgets,
                "--method", cascade_method,
                "--signature", sig,
                "--solver-timeout", str(args.solver_timeout),
                "--glasgow-bin", args.glasgow_bin,
                "--cache-dir", args.cache_dir,
                "--max-component-diag-nodes", str(args.max_component_diag_nodes),
            ]
            cmd.extend(OVERLAP_POLICY_FLAGS.get(method, []))
            if args.label_source and args.label_source != "feature":
                cmd.extend(["--label-source", args.label_source])
            if args.max_eval_queries and args.max_eval_queries > 0:
                cmd.extend(["--max-eval-queries", str(args.max_eval_queries)])
            cmd.extend(["--partial-every", "10"])
            if args.parallel_mode == "query":
                cmd.extend(["--query-workers", str(max(1, args.workers))])
            if prune_labels:
                cmd.append("--prune-query-labels")
            if component_solve:
                cmd.append("--component-solve")
                cmd.extend(["--max-component-solver-components", str(args.max_component_solver_components)])
            if not use_overlap:
                cmd.append("--no-overlap")
            if needs_models:
                for spec in args.model_specs.split(";"):
                    spec = spec.strip()
                    if spec:
                        cmd.extend(["--model", spec])
            tasks.append((tag, cmd, log_dir / f"{tag}.log"))

    print(f"[MATRIX] tasks={len(tasks)} workers={args.workers} mode={args.parallel_mode}", flush=True)
    results = []
    if args.parallel_mode == "query":
        for tag, cmd, log_path in tasks:
            res = run_command(tag, cmd, log_path, root)
            print(f"[DONE] {res}", flush=True)
            results.append(res)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = [pool.submit(run_command, tag, cmd, log_path, root) for tag, cmd, log_path in tasks]
            for fut in concurrent.futures.as_completed(futs):
                res = fut.result()
                print(f"[DONE] {res}", flush=True)
                results.append(res)

    failures = [res for res in results if not res["status"].startswith("ok")]
    if failures:
        print(f"[FAILURES] {failures}", flush=True)
        raise SystemExit(1)
    print("[MATRIX] complete", flush=True)


def shutil_which(path: str) -> str | None:
    import shutil

    found = shutil.which(path)
    if found:
        return found
    if Path(path).exists():
        return path
    return None


if __name__ == "__main__":
    main()
