"""Fail closed before a Lightning run can add redundant cloud storage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(r"C:\Users\Swastik\Desktop\CRAG\configs\compute.local.yaml")
POLICY = ROOT / "benchmarks" / "lightning_retention_policy_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-path", type=Path)
    parser.add_argument("--kind", choices=("overlay", "results", "package"))
    parser.add_argument("--required-model", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.upload_path) != bool(args.kind):
        raise SystemExit("--upload-path and --kind must be supplied together")
    if args.upload_path:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "ops" / "validate_lightning_upload.py"),
                str(args.upload_path),
                "--kind",
                args.kind,
                "--policy",
                str(POLICY),
            ],
            check=True,
        )

    accounts = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["lightning"]
    primary = [item for item in accounts if item.get("allow_new_uploads")]
    if len(accounts) != 6 or len(primary) != 1:
        raise SystemExit(
            f"account registry invalid: accounts={len(accounts)} primary={len(primary)}"
        )
    account = primary[0]
    os.environ["LIGHTNING_USER_ID"] = account["user_id"]
    os.environ["LIGHTNING_API_KEY"] = account["api_key"]

    from lightning_sdk import Teamspace

    teamspace = Teamspace(account["teamspace"], org=account["organization"])
    models = teamspace.list_models()
    names = {model.name for model in models}
    missing = sorted(set(args.required_model) - names)
    if missing:
        raise SystemExit(f"required Lightning artifacts are missing: {missing}")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    allowed_models = (
        set(policy.get("canonical_package_allowlist", []))
        | set(policy.get("protected_artifact_allowlist", []))
        | set(args.required_model)
    )
    unexpected = sorted(names - allowed_models)
    if unexpected:
        raise SystemExit(
            "unapproved Lightning artifacts must be downloaded or removed before "
            f"launch: {unexpected}"
        )
    version_bytes = 0
    version_count = 0
    max_versions = int(policy.get("max_versions_per_artifact", 1))
    excess_versions = []
    for model in models:
        versions = teamspace.list_model_versions(model.name)
        version_count += len(versions)
        version_bytes += sum(int(version.size_bytes or 0) for version in versions)
        if len(versions) > max_versions:
            excess_versions.append((model.name, len(versions)))
    if excess_versions:
        raise SystemExit(
            f"Lightning artifacts exceed the {max_versions}-version retention limit: "
            f"{excess_versions}"
        )
    max_bytes = int(policy["max_teamspace_storage_bytes"])
    if version_bytes > max_bytes:
        raise SystemExit(
            f"primary Lightning model storage {version_bytes} exceeds limit {max_bytes}; "
            "clean before launching"
        )
    print(
        f"LIGHTNING_PREFLIGHT_OK account={account['name']} models={len(models)} "
        f"versions={version_count} bytes={version_bytes} limit={max_bytes}"
    )


if __name__ == "__main__":
    main()
