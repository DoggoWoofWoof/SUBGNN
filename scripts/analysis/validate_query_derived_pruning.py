"""Validate that query-payload pruning matches the legacy planted-ID path.

This is a migration audit only.  The legacy path is evaluated for equality but
is never used to build a candidate or call a solver.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(THIS_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import benchmark_overlap_glasgow_cascade as cascade  # noqa: E402


def parse_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["dataset"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cora", "arxiv", "mag"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--hierarchy-path", default="")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--target-sizes", default="20,50,100")
    parser.add_argument(
        "--query-types",
        default="single,k_hop,degree_k_hop,multi_fine,multi_coarse,random_walk",
    )
    parser.add_argument("--seeds", default="20260607,20260608")
    parser.add_argument("--signature", required=True)
    parser.add_argument("--label-source", default="feature")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = cascade.load_named_data(args.dataset, args.data_root)
    hierarchy_path = args.hierarchy_path or cascade.default_hierarchy_path(args.dataset)
    cache_key = cascade.safe_cache_key(
        args.dataset,
        cascade.path_fingerprint(hierarchy_path),
        data.num_nodes,
        data.edge_index.size(1),
    )
    hierarchy = cascade.load_or_prepare_hierarchy(
        data, hierarchy_path, args.cache_dir, cache_key, args.dataset
    )
    hierarchy = cascade.load_or_build_overlap_index(
        data, hierarchy, args.cache_dir, cache_key
    )
    target_signatures = cascade.load_or_build_signature_tokens(
        data, args.cache_dir, cache_key
    )[args.signature]

    if args.label_source == "feature":
        target_labels = cascade.load_or_build_feature_label_tokens(
            data, args.cache_dir, cache_key
        )
    elif args.label_source.startswith("feature_bucket_"):
        bucket_count = int(args.label_source.rsplit("_", 1)[-1])
        target_labels = cascade.load_or_build_feature_bucket_label_tokens(
            data, args.cache_dir, cache_key, bucket_count
        )
    else:
        raise ValueError("migration validator currently supports feature labels/buckets")

    rows = []
    all_expectations_met = True
    for seed in [int(value) for value in parse_csv(args.seeds)]:
        queries = cascade.load_or_generate_cascade_queries(
            data,
            hierarchy,
            args.queries,
            args.target_sizes,
            seed,
            args.query_types,
            args.cache_dir,
            cache_key,
        )
        for item in queries:
            planted_ids = item["query_nodes"].detach().cpu().long()
            query = item["query"]
            legacy_labels = target_labels[planted_ids].long().tolist()
            payload_labels = cascade.derive_query_labels(query, args.label_source)
            legacy_signature_set = sorted(
                set(int(value) for value in target_signatures[planted_ids].long().tolist())
            )
            payload_signature_set = sorted(
                int(value)
                for value in cascade.derive_query_signature_tokens(
                    query, args.signature
                ).tolist()
            )
            labels_equal = legacy_labels == payload_labels
            signatures_equal = legacy_signature_set == payload_signature_set
            query_type = item.get("query_type", "")
            is_negative = bool(item.get("is_negative", False))
            if query_type in {"negative_label", "label_corrupt_negative"}:
                expectation = "payload_label_diverges_from_planted_target"
                expectation_met = not labels_equal
            elif is_negative:
                expectation = "payload_only_audit"
                expectation_met = True
            else:
                expectation = "positive_payload_matches_planted_target"
                expectation_met = labels_equal and signatures_equal
            all_expectations_met = all_expectations_met and expectation_met
            rows.append(
                {
                    "dataset": args.dataset,
                    "seed": seed,
                    "query_id": item.get("query_id", ""),
                    "query_type": query_type,
                    "target_query_size": item.get("target_query_size", 0),
                    "is_negative": is_negative,
                    "expected_match": bool(item.get("expected_match", not is_negative)),
                    "labels_equal": labels_equal,
                    "signatures_equal": signatures_equal,
                    "migration_expectation": expectation,
                    "migration_expectation_met": expectation_met,
                    "legacy_label_count": len(set(legacy_labels)),
                    "payload_label_count": len(set(payload_labels)),
                    "legacy_signature_count": len(legacy_signature_set),
                    "payload_signature_count": len(payload_signature_set),
                    "query_pruning_source": "query_payload_v1",
                }
            )

    write_rows(args.output, rows)
    print(
        f"[NO-ID VALIDATION] dataset={args.dataset} rows={len(rows)} "
        f"expectations_met={all_expectations_met} output={args.output}",
        flush=True,
    )
    if not all_expectations_met:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
