#!/usr/bin/env python3
"""Validate benchmark CSV query identities against the canonical workload."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


IDENTITY_COLUMNS = ["dataset", "seed", "query_type", "target_size", "query_index"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/canonical_workload_v1.json"),
    )
    parser.add_argument(
        "--scope",
        choices=("all", "positive", "negative"),
        default="all",
    )
    return parser.parse_args()


def normalized_column(frame, names, default=None):
    for name in names:
        if name in frame.columns:
            return frame[name]
    if default is not None:
        return pd.Series([default] * len(frame), index=frame.index)
    raise KeyError(f"missing one of columns: {', '.join(names)}")


def infer_dataset(path, manifest):
    lowered = str(path).lower()
    matches = [dataset for dataset in manifest["datasets"] if dataset in lowered]
    if len(matches) != 1:
        raise ValueError(f"cannot infer one dataset from {path}: {matches}")
    return matches[0]


def infer_seed(path):
    match = re.search(r"(?:^|[\\/_-])(?:s|seed)(20\d{6})(?:[_-]|$)", str(path))
    if not match:
        raise ValueError(f"cannot infer seed from {path}")
    return int(match.group(1))


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    allowed_types = {
        "all": manifest["positive_query_types"] + manifest["negative_query_types"],
        "positive": manifest["positive_query_types"],
        "negative": manifest["negative_query_types"],
    }[args.scope]
    expected_key = {
        "all": "unique_total_queries",
        "positive": "unique_positive_queries",
        "negative": "unique_negative_queries",
    }[args.scope]
    expected = manifest["counts_per_dataset"][expected_key]

    normalized_frames = []
    for path in args.csv:
        frame = pd.read_csv(path)
        if "dataset" in frame.columns and frame["dataset"].notna().any():
            dataset = str(frame.loc[frame["dataset"].notna(), "dataset"].iloc[0])
        else:
            dataset = infer_dataset(path, manifest)
        seed_column = next(
            (name for name in ("seed", "query_seed") if name in frame.columns), None
        )
        seed = (
            int(frame.loc[frame[seed_column].notna(), seed_column].iloc[0])
            if seed_column and frame[seed_column].notna().any()
            else infer_seed(path)
        )
        query_id = normalized_column(frame, ("query_id",))
        normalized = pd.DataFrame(
            {
                "dataset": normalized_column(frame, ("dataset",), dataset),
                "seed": normalized_column(frame, ("seed", "query_seed"), seed),
                "query_type": normalized_column(frame, ("query_type", "family")),
                "target_size": normalized_column(
                    frame, ("target_size", "target_query_size", "size")
                ),
                "query_index": query_id.astype(str).str.extract(r"_(\d+)$")[0],
            }
        )
        normalized["dataset"] = normalized["dataset"].astype(str).str.lower()
        normalized = normalized[normalized["query_type"].isin(allowed_types)]
        normalized_frames.append(normalized)

    combined = pd.concat(normalized_frames, ignore_index=True)
    if combined[IDENTITY_COLUMNS].isna().any().any():
        raise SystemExit("one or more query identities could not be normalized")

    unique = combined.drop_duplicates(IDENTITY_COLUMNS)
    failures = []
    seen_datasets = 0
    for dataset in manifest["datasets"]:
        dataset = dataset.lower()
        count = len(unique[unique["dataset"] == dataset])
        if not count:
            continue
        seen_datasets += 1
        ok = count == expected
        print(
            f"dataset={dataset} scope={args.scope} unique_queries={count} "
            f"expected={expected} source_csvs={len(args.csv)} ok={ok}"
        )
        if not ok:
            failures.append((dataset, count, expected))

    if not seen_datasets:
        failures.append(("no recognized dataset", 0, expected))

    if failures:
        raise SystemExit(f"denominator validation failed: {failures}")


if __name__ == "__main__":
    main()
