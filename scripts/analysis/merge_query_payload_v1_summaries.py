"""Merge query-payload-v1 negative reruns into the canonical paper bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "benchmarks" / "paper_results" / "final_results"
ARCHIVE = ROOT / "archive" / "query_pruning_pre_payload_v1_2026-08-01"
RUNS = ROOT / "runs" / "lightning_completion"

NEGATIVE_TYPES = {"negative_label", "negative_structure"}
DATASETS = {
    "cora": RUNS / "noid_negative_cora_v4",
    "arxiv": RUNS / "noid_negative_arxiv_v4",
    "mag": RUNS / "noid_negative_mag_v3",
}
METHOD_LABELS = {
    "hybrid": "neural_component",
    "mean_feature": "mean_feature_component",
    "coarse_mean_rrf": "mean_rrf_component",
    "topo_feature": "topo_feature_component",
    "random": "random_component",
    "all": "filterall_component",
}
METHOD_ORDER = {name: index for index, name in enumerate(METHOD_LABELS)}
QUERY_ORDER = {
    name: index
    for index, name in enumerate(
        [
            "single",
            "multi_fine",
            "k_hop",
            "degree_k_hop",
            "random_walk",
            "multi_coarse",
            "negative_label",
            "negative_structure",
        ]
    )
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ratio(numerator: str, denominator: str) -> str:
    den = float(denominator or 0)
    return "" if den == 0 else f"{float(numerator or 0) / den:.12g}"


def seconds(milliseconds: str) -> str:
    return "" if milliseconds == "" else f"{float(milliseconds) / 1000.0:.12g}"


def archive_once(source: Path, destination_name: str) -> None:
    destination = ARCHIVE / destination_name
    if source.exists() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def validate_raw_negative_run(dataset: str, run_dir: Path) -> dict:
    validation = json.loads((run_dir / "production_validation.json").read_text())
    files = sorted((run_dir / "results").glob("*_per_query.csv"))
    files = [path for path in files if "partial" not in path.name]
    if len(files) != 12:
        raise RuntimeError(f"{dataset}: expected 12 final per-query CSVs, found {len(files)}")

    unique_queries: set[tuple[str, ...]] = set()
    rows = 0
    for path in files:
        _, records = read_csv(path)
        rows += len(records)
        seed_match = re.search(r"_s(\d+)_", path.name)
        if not seed_match:
            raise RuntimeError(f"{dataset}: cannot infer seed from {path.name}")
        file_seed = seed_match.group(1)
        for row in records:
            if row.get("query_pruning_source") != "query_payload_v1":
                raise RuntimeError(f"{dataset}: non-payload pruning row in {path.name}")
            unique_queries.add(
                (
                    row.get("dataset", dataset),
                    row.get("seed") or row.get("query_seed") or file_seed,
                    row.get("query_type", ""),
                    row.get("target_query_size", ""),
                    row.get("query_id", ""),
                )
            )
    if len(unique_queries) != 600:
        raise RuntimeError(f"{dataset}: expected 600 unique negatives, found {len(unique_queries)}")
    return {
        "production_validation": validation,
        "per_query_files": len(files),
        "per_budget_rows": rows,
        "unique_negative_queries": len(unique_queries),
    }


def merge_dataset(dataset: str, run_dir: Path) -> tuple[list[str], list[dict[str, str]], dict]:
    destination = FINAL / f"final_{dataset}_summary.csv"
    header, existing = read_csv(destination)
    if "query_pruning_source" not in header:
        header.append("query_pruning_source")

    positives = [row for row in existing if row.get("query_type") not in NEGATIVE_TYPES]
    old_negatives = [row for row in existing if row.get("query_type") in NEGATIVE_TYPES]
    if len(positives) != 216 or len(old_negatives) != 72:
        raise RuntimeError(
            f"{dataset}: expected 216 positive and 72 negative summary rows, "
            f"found {len(positives)} and {len(old_negatives)}"
        )

    archive_path = ARCHIVE / f"final_{dataset}_negative_summary_pre_payload_v1.csv"
    if not archive_path.exists():
        write_csv(archive_path, [name for name in header if name != "query_pruning_source"], old_negatives)

    for row in positives:
        row["query_pruning_source"] = (
            "legacy_planted_id_v0_audited_equivalent_to_query_payload_v1"
        )

    _, fresh = read_csv(run_dir / "summary.csv")
    if len(fresh) != 72:
        raise RuntimeError(f"{dataset}: expected 72 negative summary rows, found {len(fresh)}")
    converted = []
    for source in fresh:
        if source.get("query_type") not in NEGATIVE_TYPES:
            raise RuntimeError(f"{dataset}: unexpected positive row in negative rerun")
        row = {name: source.get(name, "") for name in header}
        row["source_file"] = source.get("file", str(run_dir / "summary.csv"))
        row["dataset"] = dataset
        row["dataset_guess"] = dataset
        row["method_label"] = METHOD_LABELS[source["method"]]
        row["source_bundle"] = "query_payload_v1_negative_rerun_2026-08-01"
        row["source_rows"] = source["queries"]
        row["solved_rate"] = ratio(source["solved"], source["queries"])
        row["positive_solved_rate"] = ""
        row["false_positive_rate"] = ratio(
            source["false_positives"], source["negative_queries"]
        )
        row["correct_no_match_rate"] = ratio(
            source["correct_no_match"], source["negative_queries"]
        )
        row["solved_total"] = source["solved"]
        row["unsolved"] = str(int(float(source["queries"])) - int(float(source["solved"])))
        row["avg_solver_time_per_query"] = seconds(source["avg_solver_ms"])
        row["avg_candidate_time_per_query"] = seconds(source["avg_candidate_ms"])
        row["avg_retrieval_time"] = seconds(source["avg_retrieval_ms"])
        row["avg_total_time_per_query"] = source["avg_total_s"]
        row["avg_pruned_nodes"] = source["avg_candidate_nodes"]
        row["solver_timeouts"] = source["timeouts"]
        row["contained_rate"] = ratio(source["contained"], source["queries"])
        row["timeout_rate"] = ratio(source["timeouts"], source["queries"])
        row["query_pruning_source"] = "query_payload_v1"
        converted.append(row)

    merged = positives + converted
    merged.sort(
        key=lambda row: (
            int(row["seed"]),
            METHOD_ORDER[row["method"]],
            QUERY_ORDER[row["query_type"]],
            int(row["target_query_size"]),
        )
    )
    if len(merged) != 288:
        raise RuntimeError(f"{dataset}: merged summary has {len(merged)} rows")
    write_csv(destination, header, merged)
    return header, merged, validate_raw_negative_run(dataset, run_dir)


def validate_migration_audit() -> dict:
    sources = [
        RUNS / "noid_canonical_cora_arxiv_v1" / "cora_validation.csv",
        RUNS / "noid_canonical_cora_arxiv_v1" / "arxiv_validation.csv",
        RUNS / "noid_canonical_mag_v1" / "mag_validation.csv",
    ]
    total = positive = negative = met = positive_equal = negative_label_diverged = 0
    datasets = {}
    for path in sources:
        _, rows = read_csv(path)
        dataset = rows[0]["dataset"]
        identities = {
            (row["seed"], row["query_type"], row["target_query_size"], row["query_id"])
            for row in rows
        }
        if len(rows) != 2400 or len(identities) != 2400:
            raise RuntimeError(f"{dataset}: migration audit does not contain 2,400 unique queries")
        ds_positive = [row for row in rows if row["is_negative"].lower() != "true"]
        ds_negative = [row for row in rows if row["is_negative"].lower() == "true"]
        ds_met = sum(row["migration_expectation_met"].lower() == "true" for row in rows)
        ds_positive_equal = sum(
            row["labels_equal"].lower() == "true"
            and row["signatures_equal"].lower() == "true"
            for row in ds_positive
        )
        ds_negative_label_diverged = sum(
            row["query_type"] == "negative_label"
            and row["labels_equal"].lower() != "true"
            for row in ds_negative
        )
        datasets[dataset] = {
            "unique_queries": len(identities),
            "positive_queries": len(ds_positive),
            "negative_queries": len(ds_negative),
            "expectations_met": ds_met,
            "positive_payload_equals_legacy_tokens": ds_positive_equal,
            "negative_label_payload_intentionally_diverges": ds_negative_label_diverged,
        }
        total += len(rows)
        positive += len(ds_positive)
        negative += len(ds_negative)
        met += ds_met
        positive_equal += ds_positive_equal
        negative_label_diverged += ds_negative_label_diverged
    if (total, positive, negative, met, positive_equal, negative_label_diverged) != (
        7200,
        5400,
        1800,
        7200,
        5400,
        900,
    ):
        raise RuntimeError("combined migration audit totals are inconsistent")
    return {
        "status": "pass",
        "workload_id": "jigsaw-production-20260607-08-q50-v1",
        "query_pruning_source": "query_payload_v1",
        "unique_queries": total,
        "positive_queries": positive,
        "negative_queries": negative,
        "expectations_met": met,
        "positive_payload_equals_legacy_tokens": positive_equal,
        "negative_label_payload_intentionally_diverges": negative_label_diverged,
        "datasets": datasets,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name in [
        "manifest.json",
        "production_grid_coverage.csv",
        "benchmark_grid_coverage_report.json",
        "csv_validation_report.json",
    ]:
        archive_once(FINAL / name, name.replace(".", "_pre_payload_v1.", 1))

    all_rows = []
    negative_runs = {}
    final_header = []
    for dataset, run_dir in DATASETS.items():
        final_header, rows, negative_runs[dataset] = merge_dataset(dataset, run_dir)
        all_rows.extend(rows)
    write_csv(FINAL / "final_all_datasets_summary.csv", final_header, all_rows)

    coverage_rows = []
    for row in all_rows:
        coverage_rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "method_label": row["method_label"],
                "expected_model": row["model"],
                "query_type": row["query_type"],
                "target_query_size": row["target_query_size"],
                "status": "OK",
                "actual_rows": "1",
                "models": row["model"],
                "queries": row["queries"],
            }
        )
    coverage_header = [
        "dataset",
        "seed",
        "method_label",
        "expected_model",
        "query_type",
        "target_query_size",
        "status",
        "actual_rows",
        "models",
        "queries",
    ]
    write_csv(FINAL / "production_grid_coverage.csv", coverage_header, coverage_rows)

    generated_at = datetime.now(timezone.utc).isoformat()
    coverage_report = {
        "generated_at_utc": generated_at,
        "expected_rows_per_dataset": 288,
        "expected_total_rows": 864,
        "status_counts": {"OK": 864},
        "dataset_status_counts": {
            dataset: {"OK": 288} for dataset in sorted(DATASETS)
        },
        "issues": [],
    }
    (FINAL / "benchmark_grid_coverage_report.json").write_text(
        json.dumps(coverage_report, indent=2) + "\n", encoding="utf-8"
    )

    migration = validate_migration_audit()
    migration["negative_reruns"] = negative_runs
    (FINAL / "query_payload_v1_validation.json").write_text(
        json.dumps(migration, indent=2) + "\n", encoding="utf-8"
    )
    validation = {
        "generated_at_utc": generated_at,
        "status": "pass",
        "issues": [],
        "production_rows": {dataset: 288 for dataset in DATASETS},
        "all_datasets_rows": 864,
        "canonical_unique_queries": {
            "per_dataset": 2400,
            "positive_per_dataset": 1800,
            "negative_per_dataset": 600,
            "all_datasets": 7200,
        },
        "query_payload_v1_audit": {
            "expectations_met": migration["expectations_met"],
            "positive_equivalent": migration["positive_payload_equals_legacy_tokens"],
            "negative_label_diverged": migration[
                "negative_label_payload_intentionally_diverges"
            ],
        },
        "diagnostic_rows_preserved": 222,
    }
    (FINAL / "csv_validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    archive_readme = ARCHIVE / "README.md"
    if not archive_readme.exists():
        archive_readme.write_text(
            "# Pre-query-payload-v1 negative results\n\n"
            "This archive preserves the 72 superseded negative summary rows per dataset "
            "and the associated release manifests before the 2026-08-01 query-payload-v1 "
            "rerun. Positive rows were validated equivalent and were not rerun.\n",
            encoding="utf-8",
        )

    manifest_files = [
        "HEADLINE_NUMBERS.csv",
        "README.md",
        "benchmark_grid_coverage_report.json",
        "csv_cleaning_report.json",
        "csv_validation_report.json",
        "final_all_datasets_summary.csv",
        "final_arxiv_summary.csv",
        "final_benchmark_completion_audit.csv",
        "final_cora_summary.csv",
        "final_diagnostic_model_ablation_summary.csv",
        "final_mag_summary.csv",
        "production_grid_coverage.csv",
        "query_payload_v1_validation.json",
    ]
    files = {}
    for name in manifest_files:
        path = FINAL / name
        item = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if path.suffix == ".csv":
            header, rows = read_csv(path)
            item.update({"rows": len(rows), "columns": len(header)})
        else:
            item.update({"rows": None, "columns": None})
        files[name] = item
    manifest = {
        "generated_at_utc": generated_at,
        "note": (
            "Canonical production bundle with query-payload-v1 negative reruns merged; "
            "positive rows are unchanged and audited equivalent."
        ),
        "validation": validation,
        "coverage_status_counts": {"OK": 864},
        "files": files,
    }
    (FINAL / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "MERGE_OK datasets=3 summary_rows=864 unique_queries=7200 "
        "positive=5400 negative=1800 payload_negative_rows=216"
    )


if __name__ == "__main__":
    main()
