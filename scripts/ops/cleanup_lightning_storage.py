"""Plan, execute, and verify Lightning model/job storage cleanup."""

from __future__ import annotations

import argparse
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
OUT = ROOT / "archive" / "lightning_cleanup_2026-08-01"
CONFIRMATION = "DELETE_REDUNDANT_LIGHTNING_STORAGE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_rows(teamspace) -> list[dict]:
    rows = []
    for model in teamspace.list_models():
        versions = teamspace.list_model_versions(model.name)
        rows.append(
            {
                "name": model.name,
                "version_count": len(versions),
                "size_bytes": sum(int(version.size_bytes or 0) for version in versions),
            }
        )
    return sorted(rows, key=lambda item: item["name"])


def child() -> None:
    from lightning_sdk import Teamspace

    organization = os.environ["TARGET_ORGANIZATION"]
    teamspace_name = os.environ["TARGET_TEAMSPACE"]
    keep_models = set(json.loads(os.environ.get("KEEP_MODELS_JSON", "[]")))
    execute = os.environ.get("EXECUTE_CLEANUP") == "1"

    teamspace = Teamspace(teamspace_name, org=organization)
    project = teamspace._teamspace
    before_models = _model_rows(teamspace)
    before_jobs = sorted(
        [
            {
                "name": job.name,
                "status": str(job.status),
                "total_cost": float(job.total_cost or 0.0),
            }
            for job in teamspace.jobs
        ],
        key=lambda item: item["name"],
    )
    delete_models = [item for item in before_models if item["name"] not in keep_models]
    kept_models = [item for item in before_models if item["name"] in keep_models]
    result = {
        "captured_at": _now(),
        "organization": organization,
        "teamspace": teamspace_name,
        "execute": execute,
        "storage_before_bytes": int(project.current_storage_bytes or 0),
        "models_before": len(before_models),
        "jobs_before": len(before_jobs),
        "kept_models": kept_models,
        "delete_models": delete_models,
        "delete_model_bytes": sum(item["size_bytes"] for item in delete_models),
        "delete_jobs": before_jobs,
        "model_delete_results": [],
        "job_delete_results": [],
    }
    if not execute:
        print(json.dumps(result))
        return

    for item in delete_models:
        try:
            teamspace.delete_model(item["name"])
            outcome = "deleted"
            error = None
        except Exception as exc:  # keep cleaning independent artifacts
            outcome = "error"
            error = str(exc)[-500:]
        result["model_delete_results"].append(
            {"name": item["name"], "outcome": outcome, "error": error}
        )
        time.sleep(0.1)

    active_states = {"pending", "running", "stopping"}
    for job in list(teamspace.jobs):
        status = str(job.status).lower()
        stopped = False
        try:
            if status in active_states:
                job.stop()
                stopped = True
            job.delete()
            outcome = "deleted"
            error = None
        except Exception as exc:  # stale jobs must not block model cleanup
            outcome = "error"
            error = str(exc)[-500:]
        result["job_delete_results"].append(
            {
                "name": job.name,
                "status": status,
                "stopped": stopped,
                "outcome": outcome,
                "error": error,
            }
        )
        time.sleep(0.1)

    time.sleep(2)
    refreshed = Teamspace(teamspace_name, org=organization)
    after_models = _model_rows(refreshed)
    after_jobs = list(refreshed.jobs)
    result.update(
        {
            "completed_at": _now(),
            "storage_after_bytes": int(refreshed._teamspace.current_storage_bytes or 0),
            "models_after": len(after_models),
            "jobs_after": len(after_jobs),
            "remaining_models": after_models,
            "remaining_jobs": [job.name for job in after_jobs],
        }
    )
    print(json.dumps(result))


def _run_child(account: dict, keep_models: set[str], execute: bool) -> dict:
    env = os.environ.copy()
    env["LIGHTNING_USER_ID"] = account["user_id"]
    env["LIGHTNING_API_KEY"] = account["api_key"]
    env["TARGET_ORGANIZATION"] = account["organization"]
    env["TARGET_TEAMSPACE"] = account["teamspace"]
    env["KEEP_MODELS_JSON"] = json.dumps(sorted(keep_models))
    env["EXECUTE_CLEANUP"] = "1" if execute else "0"
    completed = subprocess.run(
        [sys.executable, __file__, "--child"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
        check=False,
    )
    if completed.returncode:
        return {
            "organization": account["organization"],
            "teamspace": account["teamspace"],
            "execute": execute,
            "fatal_error": completed.stderr.strip()[-1000:],
        }
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _keepers() -> set[str]:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    return set(policy["canonical_package_allowlist"]) | set(
        policy.get("protected_artifact_allowlist", [])
    )


def parent(execute: bool, confirmation: str) -> None:
    accounts = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["lightning"]
    if len(accounts) != 6:
        raise RuntimeError(f"Expected 6 Lightning accounts, found {len(accounts)}")
    primaries = [item for item in accounts if item.get("allow_new_uploads")]
    if len(primaries) != 1:
        raise RuntimeError(f"Expected exactly one primary account, found {len(primaries)}")
    keepers = _keepers()

    OUT.mkdir(parents=True, exist_ok=True)
    preflight = []
    for account in accounts:
        keep = keepers if account.get("allow_new_uploads") else set()
        preflight.append(_run_child(account, keep, execute=False))

    fatal = [item for item in preflight if item.get("fatal_error")]
    primary = next(
        item
        for item in preflight
        if item.get("organization") == primaries[0]["organization"]
    )
    present_primary = {item["name"] for item in primary.get("kept_models", [])}
    missing = sorted(keepers - present_primary)
    plan = {
        "captured_at": _now(),
        "account_count": len(accounts),
        "keepers": sorted(keepers),
        "missing_primary_keepers": missing,
        "fatal_preflight_errors": fatal,
        "total_storage_before_bytes": sum(
            item.get("storage_before_bytes", 0) for item in preflight
        ),
        "planned_delete_model_bytes": sum(
            item.get("delete_model_bytes", 0) for item in preflight
        ),
        "planned_delete_models": sum(len(item.get("delete_models", [])) for item in preflight),
        "planned_delete_jobs": sum(len(item.get("delete_jobs", [])) for item in preflight),
        "accounts": preflight,
    }
    (OUT / "cleanup_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(
        f"PLAN accounts={len(accounts)} models={plan['planned_delete_models']} "
        f"jobs={plan['planned_delete_jobs']} "
        f"model_gb={plan['planned_delete_model_bytes'] / 1e9:.2f}"
    )
    if fatal or missing:
        raise RuntimeError(
            f"Cleanup preflight failed: fatal={len(fatal)} missing_keepers={missing}"
        )
    if not execute:
        return
    if confirmation != CONFIRMATION:
        raise RuntimeError("Destructive cleanup confirmation string is missing")

    results = []
    for account in accounts:
        keep = keepers if account.get("allow_new_uploads") else set()
        print(f"CLEANING {account['organization']}/{account['teamspace']}")
        results.append(_run_child(account, keep, execute=True))
    execution = {
        "started_from_plan": str(OUT / "cleanup_plan.json"),
        "completed_at": _now(),
        "accounts": results,
        "deleted_models": sum(
            sum(item.get("outcome") == "deleted" for item in account.get("model_delete_results", []))
            for account in results
        ),
        "model_delete_errors": sum(
            sum(item.get("outcome") == "error" for item in account.get("model_delete_results", []))
            for account in results
        ),
        "deleted_jobs": sum(
            sum(item.get("outcome") == "deleted" for item in account.get("job_delete_results", []))
            for account in results
        ),
        "job_delete_errors": sum(
            sum(item.get("outcome") == "error" for item in account.get("job_delete_results", []))
            for account in results
        ),
        "total_storage_after_bytes": sum(
            item.get("storage_after_bytes", 0) for item in results
        ),
    }
    (OUT / "cleanup_execution.json").write_text(
        json.dumps(execution, indent=2), encoding="utf-8"
    )
    print(
        f"DONE models={execution['deleted_models']} jobs={execution['deleted_jobs']} "
        f"model_errors={execution['model_delete_errors']} "
        f"job_errors={execution['job_delete_errors']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if args.child:
        child()
    else:
        parent(args.execute, args.confirm)
