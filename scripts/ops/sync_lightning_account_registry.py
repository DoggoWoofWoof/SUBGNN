"""Synchronize authenticated Lightning accounts into gitignored YAML/TOML registries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import toml
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ops.discover_lightning_accounts import DEFAULT_ROOTS, discover_pairs


CONFIG_DIR = Path(r"C:\Users\Swastik\Desktop\CRAG\configs")
YAML_PATH = CONFIG_DIR / "compute.local.yaml"
TOML_PATH = CONFIG_DIR / "compute.local.toml"
PRIMARY_USERNAME = "swastik9895"


def probe(user_id: str, api_key: str) -> dict:
    env = os.environ.copy()
    env["LIGHTNING_USER_ID"] = user_id
    env["LIGHTNING_API_KEY"] = api_key
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("discover_lightning_accounts.py")),
            "--child",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("Lightning credential probe failed")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    accounts_by_user = {}
    for (user_id, api_key), _sources in discover_pairs(DEFAULT_ROOTS).items():
        try:
            identity = probe(user_id, api_key)
        except Exception:
            continue
        if not identity["teamspaces"]:
            continue
        authenticated_user_id = identity["user_id"]
        existing = accounts_by_user.get(authenticated_user_id)
        if existing is not None:
            continue
        project = identity["teamspaces"][0]
        username = identity["username"]
        accounts_by_user[authenticated_user_id] = {
            "name": (
                "lightning-current"
                if username == PRIMARY_USERNAME
                else f"lightning-archive-{username}"
            ),
            "user_id": authenticated_user_id,
            "api_key": api_key,
            "organization": project["organization"],
            "teamspace": project["teamspace"],
            "role": "primary" if username == PRIMARY_USERNAME else "cleanup_only",
            "allow_new_uploads": username == PRIMARY_USERNAME,
            "machine_cpu": "CPU",
            "machine_gpu": "L4",
            "source": "local-credential-discovery-2026-08-01",
        }

    accounts = sorted(
        accounts_by_user.values(),
        key=lambda item: (item["role"] != "primary", item["name"]),
    )
    if len(accounts) != 6:
        raise RuntimeError(f"Expected 6 storage-bearing Lightning accounts, found {len(accounts)}")

    yaml_payload = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    yaml_payload["lightning"] = accounts
    YAML_PATH.write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    TOML_PATH.write_text(toml.dumps({"lightning": accounts}), encoding="utf-8")
    print(f"Synchronized {len(accounts)} Lightning accounts")
    print(f"Primary upload account: {PRIMARY_USERNAME}")
    print(f"YAML: {YAML_PATH}")
    print(f"TOML: {TOML_PATH}")


if __name__ == "__main__":
    main()
