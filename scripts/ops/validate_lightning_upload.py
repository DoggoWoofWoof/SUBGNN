#!/usr/bin/env python3
"""Fail closed when a Lightning upload violates the repository retention policy."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--kind", choices=("overlay", "results", "package"), required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("benchmarks/lightning_retention_policy_v1.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if not args.path.is_dir():
        raise SystemExit(f"upload path is not a directory: {args.path}")

    files = [path for path in args.path.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    limit = int(policy["upload_limits_bytes"][args.kind])
    failures = []
    if total_bytes > limit:
        failures.append(f"size {total_bytes} exceeds {args.kind} limit {limit}")

    if args.kind == "results":
        allowed = set(policy["results_allowed_extensions"])
        forbidden = set(policy["results_forbidden_path_parts"])
        for path in files:
            relative = path.relative_to(args.path)
            lowered_parts = {part.lower() for part in relative.parts[:-1]}
            if lowered_parts & forbidden:
                failures.append(f"forbidden results directory: {relative}")
            if path.suffix.lower() not in allowed:
                failures.append(f"forbidden results file type: {relative}")

    print(
        f"upload={args.path} kind={args.kind} files={len(files)} "
        f"bytes={total_bytes} limit={limit}"
    )
    if failures:
        for failure in failures[:50]:
            print(f"ERROR: {failure}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
