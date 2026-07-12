"""Package, upload, and launch the Lightning MAG benchmark job.

This packages the current Jigsaw code plus the upgraded MAG RGCN overlap
hierarchy/model artifacts, then prints or launches a single Lightning job that
runs the production cascade matrix with internal worker parallelism.
"""

from __future__ import annotations

import argparse
import base64
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_BACKUP = ROOT / "runs/lightning_cache/kutta_v2_final_backup"
DEFAULT_PACKAGE_DIR = ROOT / "runs/lightning_mag_benchmark_package"
DEFAULT_PACKAGE_MODEL = "jigsaw-mag-rgcn-benchmark-package-v1"
DEFAULT_RESULT_MODEL = "jigsaw-mag-rgcn-benchmark-results-seed7202"
DEFAULT_CLOUD = "gcp-lightning-public-prod"

CODE_DIRS = ["src"]
CODE_FILES = [
    "requirements_lightning_rgcn.txt",
    "scripts/benchmark_glasgow.py",
    "scripts/benchmark_overlap_glasgow_cascade.py",
    "scripts/benchmark_retrieval.py",
    "scripts/retrieval_strategies.py",
    "scripts/run_mag_benchmark_matrix_local.py",
    "scripts/run_lightning_mag_benchmark.sh",
    "scripts/summarize_production_benchmarks.py",
]


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def prepare_package(args: argparse.Namespace) -> None:
    package = args.package_dir.resolve()
    artifact = args.artifact_dir.resolve()
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)

    for rel in CODE_DIRS:
        _copy_tree(ROOT / rel, package / rel)
    for rel in CODE_FILES:
        _copy_file(ROOT / rel, package / rel)

    cache = package / "cache"
    model_dir = cache / "models"
    logs_dir = cache / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    _copy_file(
        artifact / "mag_hierarchies_type_rel_2000_fine5_finecov_v1.pt",
        cache / "mag_hierarchies_type_rel_2000_fine5_finecov_v1.pt",
    )
    _copy_file(
        artifact / "mag_rgcn_final_loss_mag_seed7202_overlap_topk50_live64_checkpoint.pth",
        cache / "mag_rgcn_final_loss_mag_seed7202_overlap_topk50_live64_checkpoint.pth",
    )
    _copy_file(
        artifact / "models/mag-6_layer-model-rgcn_rgcn_final_loss_mag_seed7202_overlap_topk50_live64_best_fullcov.pth",
        model_dir / "mag-6_layer-model-rgcn_rgcn_final_loss_mag_seed7202_overlap_topk50_live64_best_fullcov.pth",
    )
    log_src = artifact / "logs/train_mag_rgcn_final_loss_mag_seed7202_overlap_topk50_live64.log"
    if log_src.exists():
        _copy_file(log_src, logs_dir / log_src.name)

    readme = package / "README_LIGHTNING_MAG_BENCHMARK.txt"
    readme.write_text(
        "\n".join(
            [
                "Jigsaw MAG RGCN benchmark package",
                "Contains upgraded overlap hierarchy, best model, final checkpoint, and benchmark scripts.",
                "Run: bash scripts/run_lightning_mag_benchmark.sh",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[PACKAGE] {package}")


def _model_ref(owner: str, teamspace: str, name: str) -> str:
    if "/" in name:
        return name
    return f"{owner}/{teamspace}/{name}"


def upload_package(args: argparse.Namespace) -> None:
    from lightning_sdk.models import upload_model

    model_ref = _model_ref(args.owner, args.teamspace, args.package_model)
    print(f"[UPLOAD] {args.package_dir} -> {model_ref}")
    upload_model(model_ref, path=args.package_dir, progress_bar=True)
    print("[UPLOAD] complete")


def build_remote_command(args: argparse.Namespace) -> str:
    package_ref = _model_ref(args.owner, args.teamspace, args.package_model)
    result_ref = _model_ref(args.owner, args.teamspace, args.result_model)
    cache_ref = _model_ref(args.owner, args.teamspace, args.query_cache_model) if args.query_cache_model else ""
    code_patch_ref = _model_ref(args.owner, args.teamspace, args.code_patch_model) if args.code_patch_model else ""
    normalizer = r'''
from pathlib import Path
for root in [Path("/workspace/jigsaw_pkg"), Path("/workspace/jigsaw_patch")]:
    if not root.exists():
        continue
    moved = 0
    for path in sorted([p for p in root.rglob("*") if "\\" in p.name], key=lambda p: len(p.parts)):
        target = path.parent.joinpath(*path.name.split("\\"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if path.is_file():
                path.unlink()
            continue
        path.replace(target)
        moved += 1
    print(f"[NORMALIZE] {root}: repaired {moved} backslash paths", flush=True)
'''
    normalizer_b64 = base64.b64encode(normalizer.encode("utf-8")).decode("ascii")
    package_validator = r'''
import base64
import shutil
from pathlib import Path

import torch
from lightning_sdk.models import download_model

package_ref = "__PACKAGE_REF__"
normalizer_code = "__NORMALIZER_B64__"
root = Path("/workspace/jigsaw_pkg")
model_paths = [
    root / "cache/models/mag-6_layer-model-rgcn_rgcn_final_loss_mag_seed7202_overlap_topk50_live64_best_fullcov.pth",
    root / "cache/mag_rgcn_final_loss_mag_seed7202_overlap_topk50_live64_checkpoint.pth",
]


def normalize_package():
    exec(base64.b64decode(normalizer_code).decode("utf-8"))


def validate_models():
    for path in model_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        torch.load(path, map_location="cpu")
        print(f"[VALIDATE] model ok: {path.name} ({path.stat().st_size} bytes)", flush=True)


last_error = None
for attempt in range(1, 4):
    try:
        validate_models()
        break
    except Exception as exc:
        last_error = exc
        print(f"[VALIDATE] package model check failed attempt {attempt}: {exc}", flush=True)
        if attempt >= 3:
            raise
        shutil.rmtree(root, ignore_errors=True)
        download_model(package_ref, str(root), progress_bar=True)
        normalize_package()
else:
    raise RuntimeError(last_error)
'''
    package_validator = package_validator.replace("__PACKAGE_REF__", package_ref).replace(
        "__NORMALIZER_B64__", normalizer_b64
    )
    package_validator_b64 = base64.b64encode(package_validator.encode("utf-8")).decode("ascii")
    build_tool_setup = (
        "if command -v apt-get >/dev/null 2>&1; then "
        "apt-get update -qq && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++; "
        "fi; "
    )
    if args.run_mode != "query-cache":
        build_tool_setup = (
            "if command -v apt-get >/dev/null 2>&1; then "
            "apt-get update -qq && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential g++ cmake git libboost-all-dev libgmp-dev; "
            "fi; "
        )
    inner = (
        "set -euo pipefail; "
        "python -m pip install -q -U lightning-sdk; "
        f"python -c \"from lightning_sdk.models import download_model; "
        f"download_model('{package_ref}', '/workspace/jigsaw_pkg', progress_bar=True)\"; "
        f"python -c \"import base64; exec(base64.b64decode('{normalizer_b64}').decode())\"; "
        + (
            f"python -c \"import base64; exec(base64.b64decode('{package_validator_b64}').decode())\"; "
            if args.run_mode == "benchmark"
            else ""
        )
        + (
            f"python -c \"from lightning_sdk.models import download_model; "
            f"download_model('{code_patch_ref}', '/workspace/jigsaw_patch', progress_bar=True)\"; "
            f"python -c \"import base64; exec(base64.b64decode('{normalizer_b64}').decode())\"; "
            "cp -a /workspace/jigsaw_patch/. /workspace/jigsaw_pkg/; "
            if code_patch_ref
            else ""
        )
        + "cd /workspace/jigsaw_pkg; "
        + "chmod +x scripts/run_lightning_mag_benchmark.sh; "
        + build_tool_setup
        + "python -m pip install -q matplotlib; "
        f"export LIGHTNING_RESULTS_MODEL='{result_ref}'; "
        f"export RUN_MODE='{args.run_mode}'; "
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
        f"export SOLVER_TIMEOUT='{args.solver_timeout}'; "
        f"export JIGSAW_TORCH_THREADS='{args.torch_threads}'; "
        f"export CACHE_ENCODE_BATCH_SIZE='{args.cache_encode_batch_size}'; "
        f"export MAX_EVAL_QUERIES='{args.max_eval_queries}'; "
        f"export LABEL_SOURCE='{args.label_source}'; "
        "bash scripts/run_lightning_mag_benchmark.sh"
    )
    return "bash -lc " + shlex.quote(inner)


def print_job_command(args: argparse.Namespace) -> None:
    command = build_remote_command(args)
    env_flags = ""
    if os.environ.get("LIGHTNING_USER_ID"):
        env_flags += ' -e "LIGHTNING_USER_ID=$env:LIGHTNING_USER_ID"'
    if os.environ.get("LIGHTNING_API_KEY"):
        env_flags += ' -e "LIGHTNING_API_KEY=$env:LIGHTNING_API_KEY"'
    print(
        "$cmd = @'\n"
        f"{command}\n"
        "'@\n"
        ".\\.venv_modal\\Scripts\\python.exe scripts\\lightning_cli_windows.py job run "
        f"--name {args.job_name} "
        f"--machine {args.machine} "
        f"--user {args.owner} "
        f"--teamspace {args.teamspace} "
        + (f"--cloud {args.cloud} " if args.cloud else "")
        + "--image pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime "
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
        "--user",
        args.owner,
        "--teamspace",
        args.teamspace,
    ]
    if args.cloud:
        cmd.extend(["--cloud", args.cloud])
    cmd.extend([
        "--image",
        "pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime",
        "--command",
        command,
    ])
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
    parser.add_argument("--owner", default=os.environ.get("LIGHTNING_OWNER", "kuttakamina9895"))
    parser.add_argument("--teamspace", default=os.environ.get("LIGHTNING_TEAMSPACE", "deploy-model-project"))
    parser.add_argument("--package-model", default=DEFAULT_PACKAGE_MODEL)
    parser.add_argument("--result-model", default=DEFAULT_RESULT_MODEL)
    parser.add_argument("--job-name", default="jigsaw-mag-rgcn-prod-benchmark")
    parser.add_argument("--machine", default="T4")
    parser.add_argument("--cloud", default=os.environ.get("LIGHTNING_CLOUD", DEFAULT_CLOUD))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--parallel-mode", choices=["task", "query"], default="task")
    parser.add_argument("--run-mode", choices=["query-cache", "benchmark"], default="benchmark")
    parser.add_argument("--query-cache-model", default="")
    parser.add_argument("--code-patch-model", default="")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--target-sizes", default="20,50,100")
    parser.add_argument("--query-types", default="all")
    parser.add_argument("--seeds", default="20260607,20260608")
    parser.add_argument(
        "--methods",
        default="neural_component,random_component,mean_feature_component,mean_rrf_component,filterall_component",
    )
    parser.add_argument("--ablation-set", choices=["none", "full"], default="none")
    parser.add_argument("--budgets", default="20,50,100,200,500,1000")
    parser.add_argument("--solver-timeout", default="5")
    parser.add_argument("--torch-threads", default="4")
    parser.add_argument("--cache-encode-batch-size", default="128")
    parser.add_argument("--max-eval-queries", default="0")
    parser.add_argument("--label-source", choices=["feature", "class"], default="feature")
    parser.add_argument("--interruptible", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    package = sub.add_parser("prepare-package")
    package.add_argument("--artifact-dir", type=Path, default=DEFAULT_FINAL_BACKUP)
    package.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
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
