"""Package, upload, and launch generic Lightning production benchmark jobs.

This is the dataset-generic companion to ``lightning_mag_benchmark.py``.  It
keeps the same Lightning result-resume and periodic-upload workflow, but lets
callers choose Cora/Arxiv/MAG settings without hardcoded MAG model paths.
"""

from __future__ import annotations

import argparse
import base64
import os
import shlex
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_DIR = ROOT / "runs/lightning_production_benchmark_package"
DEFAULT_PACKAGE_MODEL = "jigsaw-production-benchmark-package-v1"
DEFAULT_RESULT_MODEL = "jigsaw-production-benchmark-results-v1"
DEFAULT_CLOUD = "gcp-lightning-public-prod"

CODE_DIRS = ["src"]
CODE_FILES = [
    "requirements_lightning_rgcn.txt",
    "scripts/benchmark_glasgow.py",
    "scripts/benchmark_overlap_glasgow_cascade.py",
    "scripts/benchmark_retrieval.py",
    "scripts/retrieval_strategies.py",
    "scripts/run_mag_benchmark_matrix_local.py",
    "scripts/launchers/run_lightning_mag_benchmark.sh",
    "scripts/summarize_production_benchmarks.py",
]


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _model_ref(owner: str, teamspace: str, name: str) -> str:
    if "/" in name:
        return name
    return f"{owner}/{teamspace}/{name}"


def prepare_package(args: argparse.Namespace) -> None:
    package = args.package_dir.resolve()
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)

    for rel in CODE_DIRS:
        _copy_tree(ROOT / rel, package / rel)
    for rel in CODE_FILES:
        _copy_file(ROOT / rel, package / rel)

    cache = package / "cache"
    model_dir = cache / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for rel in args.include_data:
        _copy_tree(ROOT / rel, package / rel)

    # Common local artifacts used by the non-MAG completion jobs.
    default_artifacts = [
        (ROOT / "models/cora-6_layer-model-jigsaw.pth", model_dir / "cora-6_layer-model-jigsaw.pth"),
        (
            ROOT / "runs/overlap_models/cora/models/cora-6_layer-model-graphsage_graphsage_final_loss_cora_seed7202_overlap_topk10_live20_best_fullcov.pth",
            model_dir / "cora-6_layer-model-graphsage_graphsage_final_loss_cora_seed7202_overlap_topk10_live20_best_fullcov.pth",
        ),
        (
            ROOT / "runs/overlap_models/cora/models/cora-6_layer-model-graphsage_graphsage_final_loss_cora_seed7202_overlap_topk10_live20.pth",
            model_dir / "cora-6_layer-model-graphsage_graphsage_final_loss_cora_seed7202_overlap_topk10_live20.pth",
        ),
        (
            ROOT / "models/arxiv-6_layer-model-jigsaw_coverage_final_ablation_final_seed7101_best_fullcov.pth",
            model_dir / "arxiv-6_layer-model-jigsaw_coverage_final_ablation_final_seed7101_best_fullcov.pth",
        ),
        (
            ROOT / "models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth",
            model_dir / "arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth",
        ),
        (
            ROOT / "models/arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6.pth",
            model_dir / "arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6.pth",
        ),
        (
            ROOT / "runs/overlap_models/arxiv/models/arxiv-6_layer-model-graphsage_graphsage_final_loss_arxiv_seed7202_overlap_topk50_live64_best_fullcov.pth",
            model_dir / "arxiv-6_layer-model-graphsage_graphsage_final_loss_arxiv_seed7202_overlap_topk50_live64_best_fullcov.pth",
        ),
        (
            ROOT / "runs/overlap_models/arxiv/models/arxiv-6_layer-model-graphsage_graphsage_final_loss_arxiv_seed7202_overlap_topk50_live64.pth",
            model_dir / "arxiv-6_layer-model-graphsage_graphsage_final_loss_arxiv_seed7202_overlap_topk50_live64.pth",
        ),
        (
            ROOT / "runs/migration/fair_ablation/arxiv_hierarchies_finecov_v1.pt",
            cache / "arxiv_hierarchies_finecov_v1.pt",
        ),
        (
            ROOT / "runs/overlap_models/cora/cora_hierarchies_finecov_v1.pt",
            cache / "cora_hierarchies_finecov_v1.pt",
        ),
        (
            ROOT / "runs/overlap_models/arxiv/arxiv_hierarchies_finecov_v1.pt",
            cache / "arxiv_hierarchies_finecov_v1.pt",
        ),
    ]
    for src, dst in default_artifacts:
        if src.exists():
            _copy_file(src, dst)

    for spec in args.copy_file:
        if "=" not in spec:
            raise ValueError(f"--copy-file must be src=dest, got {spec}")
        src, dst = spec.split("=", 1)
        _copy_file(ROOT / src, package / dst)

    (package / "README_LIGHTNING_PRODUCTION_BENCHMARK.txt").write_text(
        "\n".join(
            [
                "Generic Jigsaw production benchmark package",
                "Contains benchmark scripts plus local Cora/Arxiv artifacts when available.",
                "Run: bash scripts/launchers/run_lightning_mag_benchmark.sh with DATASET/METHODS env vars.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[PACKAGE] {package}")


def upload_package(args: argparse.Namespace) -> None:
    from lightning_sdk.models import upload_model

    model_ref = _model_ref(args.owner, args.teamspace, args.package_model)
    print(f"[UPLOAD] {args.package_dir} -> {model_ref}")
    upload_model(model_ref, path=args.package_dir, progress_bar=True)
    print("[UPLOAD] complete")


def _normalizer_b64() -> str:
    code = r'''
from pathlib import Path
for root in [Path("/workspace/jigsaw_pkg")]:
    if not root.exists():
        continue
    moved = 0
    for path in sorted([p for p in root.rglob("*") if "\\" in p.name], key=lambda p: len(p.parts)):
        target = path.parent.joinpath(*[part for part in path.name.split("\\") if part])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if path.is_file():
                path.unlink()
            continue
        path.replace(target)
        moved += 1
    print(f"[NORMALIZE] {root}: repaired {moved} backslash paths", flush=True)
'''
    return base64.b64encode(code.encode("utf-8")).decode("ascii")


def _partial_safe_patch_b64() -> str:
    code = r"""
from pathlib import Path

summary = Path("/workspace/jigsaw_pkg/scripts/summarize_production_benchmarks.py")
if summary.exists():
    text = summary.read_text(encoding="utf-8")
    if "def is_partial_per_query_path" not in text:
        old = '''def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = []
    for path in args.csvs:
        matches = glob.glob(path)
        for resolved in (matches if matches else [path]):
            rows.extend(summarize_file(resolved))
'''
        new = '''def is_partial_per_query_path(path):
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
'''
        if old not in text:
            raise RuntimeError("Could not apply partial-safe summarizer patch")
        summary.write_text(text.replace(old, new), encoding="utf-8")

runner = Path("/workspace/jigsaw_pkg/scripts/launchers/run_lightning_mag_benchmark.sh")
if runner.exists():
    text = runner.read_text(encoding="utf-8")
    old = '''python scripts/summarize_production_benchmarks.py \\
  "$RESULT_ROOT/results/*_per_query.csv" \\
  --output "$RESULT_ROOT/mag_rgcn_production_summary.csv" || true
'''
    new = '''mapfile -t FINAL_PER_QUERY_CSVS < <(
  find "$RESULT_ROOT/results" -maxdepth 1 -type f \\
    -name '*_per_query.csv' ! -name '*_partial_per_query.csv' | sort
)
if [[ "${#FINAL_PER_QUERY_CSVS[@]}" -gt 0 ]]; then
  python scripts/summarize_production_benchmarks.py \\
    "${FINAL_PER_QUERY_CSVS[@]}" \\
    --output "$RESULT_ROOT/mag_rgcn_production_summary.csv" || true
else
  echo "[WARN] No final per-query CSVs found for summary" >&2
fi
'''
    if old in text:
        runner.write_text(text.replace(old, new), encoding="utf-8")

print("[PATCH] partial-safe summary patch ready", flush=True)
"""
    return base64.b64encode(code.encode("utf-8")).decode("ascii")


def build_remote_command(args: argparse.Namespace) -> str:
    package_ref = _model_ref(args.owner, args.teamspace, args.package_model)
    result_ref = _model_ref(args.owner, args.teamspace, args.result_model)
    cache_ref = _model_ref(args.owner, args.teamspace, args.query_cache_model) if args.query_cache_model else ""
    code_patch_ref = _model_ref(args.owner, args.teamspace, args.code_patch_model) if args.code_patch_model else ""
    normalizer_b64 = _normalizer_b64()
    partial_safe_patch_b64 = _partial_safe_patch_b64()

    build_tool_setup = (
        "if command -v apt-get >/dev/null 2>&1; then "
        "apt-get update -qq && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential g++ cmake git libboost-all-dev libgmp-dev; "
        "fi; "
    )
    code_patch_setup = ""
    if code_patch_ref:
        code_patch_setup = (
            f"python -c \"from lightning_sdk.models import download_model; download_model('{code_patch_ref}', '/workspace/jigsaw_patch', progress_bar=True)\"; "
            "cp -a /workspace/jigsaw_patch/. /workspace/jigsaw_pkg/; "
            # Normalize AFTER the overlay copy: Windows uploads store nested paths with
            # backslashes, so the patch lands as flat 'scripts\\launchers\\x.sh' files until
            # this repairs them into real nested dirs under /workspace/jigsaw_pkg.
            f"python -c \"import base64; exec(base64.b64decode('{normalizer_b64}').decode())\"; "
        )
    inner = (
        "set -euo pipefail; "
        "python -m pip install -q -U lightning-sdk; "
        f"python -c \"from lightning_sdk.models import download_model; download_model('{package_ref}', '/workspace/jigsaw_pkg', progress_bar=True)\"; "
        f"python -c \"import base64; exec(base64.b64decode('{normalizer_b64}').decode())\"; "
        f"python -c \"import base64; exec(base64.b64decode('{partial_safe_patch_b64}').decode())\"; "
        + code_patch_setup
        + "cd /workspace/jigsaw_pkg; "
        "chmod +x scripts/launchers/run_lightning_mag_benchmark.sh; "
        + build_tool_setup
        + "python -m pip install -q matplotlib; "
        f"export LIGHTNING_RESULTS_MODEL='{result_ref}'; "
        f"export RUN_MODE='{args.run_mode}'; "
        f"export DATASET='{args.dataset}'; "
        f"export DATA_ROOT='{args.data_root}'; "
        f"export HIERARCHY_PATH='{args.hierarchy_path}'; "
        f"export MODEL_SPECS='{args.model_specs}'; "
        f"export LIGHTNING_QUERY_CACHE_MODEL='{cache_ref}'; "
        f"export CASCADE_WORKERS='{args.workers}'; "
        f"export CASCADE_PARALLEL_MODE='{args.parallel_mode}'; "
        f"export QUERIES_PER_TYPE='{args.queries}'; "
        f"export TARGET_SIZES='{args.target_sizes}'; "
        f"export QUERY_TYPES='{args.query_types}'; "
        f"export SEEDS='{args.seeds}'; "
        f"export METHODS='{args.methods}'; "
        f"export ABLATION_SET='{args.ablation_set}'; "
        f"export BUDGETS='{args.budgets}'; "
        f"export FULL_BUDGET='{args.full_budget}'; "
        f"export SIGNATURE='{args.signature}'; "
        f"export SOLVER_TIMEOUT='{args.solver_timeout}'; "
        f"export OUTPUT_PREFIX='{args.output_prefix}'; "
        f"export JIGSAW_TORCH_THREADS='{args.torch_threads}'; "
        f"export CACHE_ENCODE_BATCH_SIZE='{args.cache_encode_batch_size}'; "
        f"export MAX_EVAL_QUERIES='{args.max_eval_queries}'; "
        + "".join(f"export {kv}; " for kv in (getattr(args, "extra_env", None) or []))
        + f"bash {getattr(args, 'run_script', None) or 'scripts/launchers/run_lightning_mag_benchmark.sh'}"
    )
    return "bash -lc " + shlex.quote(inner)


def print_job_command(args: argparse.Namespace) -> None:
    command = build_remote_command(args)
    env_flags = ""
    if os.environ.get("LIGHTNING_USER_ID"):
        env_flags += ' -e "LIGHTNING_USER_ID=$env:LIGHTNING_USER_ID"'
    if os.environ.get("LIGHTNING_API_KEY"):
        env_flags += ' -e "LIGHTNING_API_KEY=$env:LIGHTNING_API_KEY"'
    owner_flag = "--org" if args.owner_kind == "org" else "--user"
    print(
        "$cmd = @'\n"
        f"{command}\n"
        "'@\n"
        ".\\.venv_modal\\Scripts\\python.exe scripts\\lightning_cli_windows.py job run "
        f"--name {args.job_name} "
        f"--machine {args.machine} "
        f"{owner_flag} {args.owner} "
        f"--teamspace {args.teamspace} "
        + (f"--cloud {args.cloud} " if args.cloud else "")
        + f"--image {args.image} "
        f"--command $cmd{env_flags}"
        + (" --interruptible" if args.interruptible else "")
    )


def launch_job(args: argparse.Namespace) -> None:
    command = build_remote_command(args)
    cli = ROOT / ".venv_modal/Scripts/python.exe"
    shim = ROOT / "scripts/lightning_cli_windows.py"
    cmd = [
        str(cli),
        str(shim),
        "job",
        "run",
        "--name",
        args.job_name,
        "--machine",
        args.machine,
        "--org" if args.owner_kind == "org" else "--user",
        args.owner,
        "--teamspace",
        args.teamspace,
    ]
    if args.cloud:
        cmd.extend(["--cloud", args.cloud])
    cmd.extend(["--image", args.image, "--command", command])
    if args.interruptible:
        cmd.append("--interruptible")
    if os.environ.get("LIGHTNING_USER_ID"):
        cmd.extend(["-e", f"LIGHTNING_USER_ID={os.environ['LIGHTNING_USER_ID']}"])
    if os.environ.get("LIGHTNING_API_KEY"):
        cmd.extend(["-e", f"LIGHTNING_API_KEY={os.environ['LIGHTNING_API_KEY']}"])
    redacted = []
    skip_value = False
    for item in cmd:
        if skip_value:
            redacted.append("<redacted-env>")
            skip_value = False
            continue
        redacted.append(item)
        if item == "-e":
            skip_value = True
    print("[LAUNCH]", " ".join(redacted))
    subprocess.run(cmd, cwd=ROOT, check=True)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--owner", default=os.environ.get("LIGHTNING_OWNER", "whenthedarknightrises"))
    parser.add_argument("--owner-kind", choices=["user", "org"], default=os.environ.get("LIGHTNING_OWNER_KIND", "user"))
    parser.add_argument("--teamspace", default=os.environ.get("LIGHTNING_TEAMSPACE", "financial-llm-training-project"))
    parser.add_argument("--package-model", default=DEFAULT_PACKAGE_MODEL)
    parser.add_argument("--code-patch-model", default="")
    parser.add_argument("--result-model", default=DEFAULT_RESULT_MODEL)
    parser.add_argument("--job-name", default="jigsaw-production-benchmark")
    parser.add_argument("--machine", default="CPU_X_8")
    parser.add_argument("--cloud", default=os.environ.get("LIGHTNING_CLOUD", DEFAULT_CLOUD))
    parser.add_argument("--image", default="pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--hierarchy-path", default="")
    parser.add_argument("--model-specs", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-mode", choices=["task", "query"], default="task")
    parser.add_argument("--run-mode", choices=["query-cache", "benchmark"], default="benchmark")
    parser.add_argument("--query-cache-model", default="")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--target-sizes", default="20,50,100")
    parser.add_argument("--query-types", default="all")
    parser.add_argument("--seeds", default="20260607,20260608")
    parser.add_argument("--methods", default="neural_component,random_component,mean_feature_component,mean_rrf_component,topo_feature_component,filterall_component")
    parser.add_argument("--ablation-set", choices=["none", "full"], default="none")
    parser.add_argument("--budgets", default="2,5,10,20")
    parser.add_argument("--full-budget", default="20")
    parser.add_argument("--signature", default="type_feat32")
    parser.add_argument("--solver-timeout", default="5")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--torch-threads", default="4")
    parser.add_argument("--cache-encode-batch-size", default="128")
    parser.add_argument("--max-eval-queries", default="0")
    parser.add_argument("--interruptible", action="store_true")
    parser.add_argument("--run-script", default="scripts/launchers/run_lightning_mag_benchmark.sh",
                        help="Remote shell to run (override for training, e.g. scripts/launchers/run_lightning_jigsaw_train.sh)")
    parser.add_argument("--extra-env", nargs="*", default=[],
                        help="Extra KEY=VAL env exports injected before the run script (e.g. DATASET=cora EPOCHS=1)")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    package = sub.add_parser("prepare-package")
    package.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    package.add_argument("--include-data", action="append", default=["data/Cora"])
    package.add_argument("--copy-file", action="append", default=[])
    package.set_defaults(func=prepare_package)

    upload = sub.add_parser("upload-package")
    upload.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    add_common(upload)
    upload.set_defaults(func=upload_package)

    command = sub.add_parser("job-command")
    add_common(command)
    command.set_defaults(func=print_job_command)

    launch = sub.add_parser("launch-job")
    add_common(launch)
    launch.set_defaults(func=launch_job)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
