"""CLI helper for Lightning VM/JOBS MAG RGCN scratch training.

Modes:
  vm-upload-run: upload prepared package to an existing VM and start training.
  job-command: print a Docker Job command that downloads the package model.

This intentionally starts from scratch. The only reused artifact is the
code package; the overlap-aware MAG hierarchy is regenerated on the worker.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path


DEFAULT_OWNER = "swastik9895"
DEFAULT_TEAMSPACE = "financial-llm-training-project"
DEFAULT_PACKAGE_MODEL = "jigsaw-rgcn-code-package-v2"
DEFAULT_PACKAGE = Path("runs/lightning_rgcn_code_package")


def vm_upload_run(args: argparse.Namespace) -> None:
    from lightning_sdk.studio import VM

    vm = VM(name=args.vm_name, teamspace="general", org="pes1ug23cs622-org", create_ok=False)
    print(f"[VM] {vm.name} status={vm.status}")
    if str(vm.status).lower() != "running":
        vm.start()
        print(f"[VM] started {vm.name} status={vm.status}")

    print(f"[UPLOAD] {args.package} -> {args.remote_path}")
    vm.upload_folder(str(args.package), remote_path=args.remote_path)

    cmd = (
        f"cd ~/{args.remote_path} && "
        f"nohup bash scripts/run_lightning_rgcn_mag.sh "
        f"> rgcn_mag_scratch.log 2>&1 & echo $!"
    )
    print("[RUN]", cmd)
    out = vm.run(cmd)
    print(out)


def print_job_command(args: argparse.Namespace) -> None:
    # User/API key must be supplied by caller env if this path is used:
    #   -e LIGHTNING_USER_ID=...
    #   -e LIGHTNING_API_KEY=...
    model_ref = args.package_model
    if "/" not in model_ref:
        model_ref = f"{args.owner}/{args.teamspace}/{model_ref}"
    normalizer = r'''
from pathlib import Path

root = Path("/workspace/jigsaw_pkg")
moved = 0
for path in sorted(
    [candidate for candidate in root.rglob("*") if "\\" in candidate.name],
    key=lambda candidate: len(candidate.parts),
):
    target = path.parent.joinpath(*path.name.split("\\"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if path.is_file():
            path.unlink()
        continue
    path.replace(target)
    moved += 1
print(f"[NORMALIZE] repaired {moved} backslash paths", flush=True)
'''
    normalizer_b64 = base64.b64encode(normalizer.encode("utf-8")).decode("ascii")
    command = (
        "python -m pip install -q -U lightning-sdk && "
        f"python -c \"from lightning_sdk.models import download_model; "
        f"download_model('{model_ref}', '/workspace/jigsaw_pkg', progress_bar=True)\" && "
        f"python -c \"import base64; exec(base64.b64decode('{normalizer_b64}').decode())\" && "
        "find /workspace/jigsaw_pkg -maxdepth 4 -type f -print && "
        "RUNNER=$(find /workspace/jigsaw_pkg -path \"*/scripts/run_lightning_rgcn_mag.sh\" -print -quit) && "
        "echo \"[RUNNER] $RUNNER\" && "
        "test -n \"$RUNNER\" && "
        "cd \"$(dirname \"$RUNNER\")/..\" && "
        "mkdir -p cache && "
        "CACHE_ROOT=\"$PWD/cache\" bash scripts/run_lightning_rgcn_mag.sh"
    )
    print(
        "$cmd = @'\n"
        f"{command}\n"
        "'@\n"
        ".\\.venv_modal\\Scripts\\python.exe scripts\\lightning_cli_windows.py job run "
        f"--name {args.job_name} "
        f"--machine {args.machine} "
        f"--user {args.owner} "
        f"--teamspace {args.teamspace} "
        "--image pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime "
        "--command $cmd"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    vm = sub.add_parser("vm-upload-run")
    vm.add_argument("--vm-name", default="jigsaw-rgcn-mag")
    vm.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    vm.add_argument("--remote-path", default="jigsaw_rgcn_mag")
    vm.set_defaults(func=vm_upload_run)

    job = sub.add_parser("job-command")
    job.add_argument("--job-name", default="jigsaw-rgcn-mag-scratch-overlap")
    job.add_argument("--machine", default="L4")
    job.add_argument("--owner", default=DEFAULT_OWNER)
    job.add_argument("--teamspace", default=DEFAULT_TEAMSPACE)
    job.add_argument("--package-model", default=DEFAULT_PACKAGE_MODEL)
    job.set_defaults(func=print_job_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
