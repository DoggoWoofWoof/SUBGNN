"""Resume a verified Lightning cleanup with bounded concurrent API calls."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(r"C:\Users\Swastik\Desktop\CRAG\configs\compute.local.yaml")
POLICY = ROOT / "benchmarks" / "lightning_retention_policy_v1.json"
PLAN = ROOT / "archive" / "lightning_cleanup_2026-08-01" / "cleanup_plan.json"
OUTPUT = ROOT / "archive" / "lightning_cleanup_2026-08-01" / "cleanup_execution_fast.json"
CONFIRMATION = "DELETE_REDUNDANT_LIGHTNING_STORAGE"


def keepers() -> set[str]:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    return set(policy["canonical_package_allowlist"]) | set(
        policy.get("protected_artifact_allowlist", [])
    )


def child() -> None:
    from lightning_sdk import Teamspace

    teamspace = Teamspace(
        os.environ["TARGET_TEAMSPACE"], org=os.environ["TARGET_ORGANIZATION"]
    )
    keep = set(json.loads(os.environ.get("KEEP_MODELS_JSON", "[]")))
    model_names = sorted(model.name for model in teamspace.list_models())
    delete_names = [name for name in model_names if name not in keep]

    def delete_model(name: str) -> dict:
        try:
            teamspace.delete_model(name)
            return {"name": name, "outcome": "deleted", "error": None}
        except Exception as exc:
            return {"name": name, "outcome": "error", "error": str(exc)[-500:]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        model_results = list(pool.map(delete_model, delete_names))

    jobs = list(teamspace.jobs)

    def delete_job(job) -> dict:
        status = str(job.status).lower()
        try:
            if status in {"pending", "running", "stopping"}:
                job.stop()
            job.delete()
            return {"name": job.name, "outcome": "deleted", "error": None}
        except Exception as exc:
            return {"name": job.name, "outcome": "error", "error": str(exc)[-500:]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        job_results = list(pool.map(delete_job, jobs))

    time.sleep(3)
    refreshed = Teamspace(
        os.environ["TARGET_TEAMSPACE"], org=os.environ["TARGET_ORGANIZATION"]
    )
    print(
        json.dumps(
            {
                "organization": os.environ["TARGET_ORGANIZATION"],
                "teamspace": os.environ["TARGET_TEAMSPACE"],
                "model_results": model_results,
                "job_results": job_results,
                "remaining_models": sorted(model.name for model in refreshed.list_models()),
                "remaining_jobs": sorted(job.name for job in refreshed.jobs),
                "storage_after_bytes": int(
                    refreshed._teamspace.current_storage_bytes or 0
                ),
            }
        )
    )


def run_account(account: dict, keep: set[str]) -> dict:
    env = os.environ.copy()
    env["LIGHTNING_USER_ID"] = account["user_id"]
    env["LIGHTNING_API_KEY"] = account["api_key"]
    env["TARGET_ORGANIZATION"] = account["organization"]
    env["TARGET_TEAMSPACE"] = account["teamspace"]
    env["KEEP_MODELS_JSON"] = json.dumps(sorted(keep))
    completed = subprocess.run(
        [sys.executable, __file__, "--child"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    if completed.returncode:
        return {
            "organization": account["organization"],
            "teamspace": account["teamspace"],
            "fatal_error": completed.stderr.strip()[-1000:],
        }
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main(confirmation: str) -> None:
    if confirmation != CONFIRMATION:
        raise RuntimeError("Destructive cleanup confirmation string is missing")
    if not PLAN.exists():
        raise RuntimeError("Verified cleanup plan is missing")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("fatal_preflight_errors") or plan.get("missing_primary_keepers"):
        raise RuntimeError("Verified cleanup plan contains failed preconditions")

    accounts = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["lightning"]
    required = keepers()
    primary = next(item for item in accounts if item.get("allow_new_uploads"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(
                run_account,
                account,
                required if account.get("allow_new_uploads") else set(),
            ): account["name"]
            for account in accounts
        }
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"CLEANED {futures[future]} remaining_models="
                f"{len(result.get('remaining_models', []))} "
                f"remaining_jobs={len(result.get('remaining_jobs', []))}"
            )

    primary_result = next(
        item for item in results if item.get("organization") == primary["organization"]
    )
    missing = sorted(required - set(primary_result.get("remaining_models", [])))
    unexpected = {
        f"{item.get('organization')}/{item.get('teamspace')}": item.get(
            "remaining_models", []
        )
        for item in results
        if item.get("organization") != primary["organization"]
        and item.get("remaining_models")
    }
    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_plan": str(PLAN),
        "results": results,
        "missing_primary_keepers": missing,
        "unexpected_archive_models": unexpected,
        "fatal_errors": [item for item in results if item.get("fatal_error")],
        "model_delete_errors": sum(
            sum(row["outcome"] == "error" for row in item.get("model_results", []))
            for item in results
        ),
        "job_delete_errors": sum(
            sum(row["outcome"] == "error" for row in item.get("job_results", []))
            for item in results
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"VERIFY missing_keepers={len(missing)} unexpected_archives={len(unexpected)} "
        f"fatal={len(report['fatal_errors'])} "
        f"model_errors={report['model_delete_errors']} "
        f"job_errors={report['job_delete_errors']}"
    )
    if missing or unexpected or report["fatal_errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    child() if args.child else main(args.confirm)
