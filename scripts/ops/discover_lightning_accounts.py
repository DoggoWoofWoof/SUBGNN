"""Discover locally recorded Lightning credentials without printing secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOTS = (
    Path(r"C:\Users\Swastik\Desktop\CRAG"),
    Path(r"C:\Users\Swastik\Desktop\Jigsaw"),
    Path(r"C:\Users\Swastik\.codex"),
)
UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
PAIR_PATTERNS = (
    re.compile(
        rf"LIGHTNING_USER_ID\s*[=:]\s*['\"]?({UUID}).{{0,1000}}?"
        rf"LIGHTNING_API_KEY\s*[=:]\s*['\"]?({UUID})",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"user_id\s*:\s*['\"]?({UUID}).{{0,500}}?"
        rf"api_key\s*:\s*['\"]?({UUID})",
        re.IGNORECASE | re.DOTALL,
    ),
)


def _sanitize_error(value: str) -> str:
    return re.sub(UUID, "<uuid>", value)[-500:]


def _credential_files(roots: tuple[Path, ...]) -> list[Path]:
    command = [
        "rg",
        "--hidden",
        "-l",
        "-g",
        "*.yaml",
        "-g",
        "*.yml",
        "-g",
        "*.json",
        "-g",
        "*.jsonl",
        "-g",
        "*.toml",
        "-g",
        "*.py",
        "-g",
        "*.ps1",
        "-g",
        "*.sh",
        "-g",
        "*.txt",
        "LIGHTNING_API_KEY|api_key\\s*:",
        *(str(root) for root in roots if root.exists()),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return sorted({Path(line) for line in completed.stdout.splitlines() if line})


def discover_pairs(roots: tuple[Path, ...]) -> dict[tuple[str, str], set[str]]:
    pairs: dict[tuple[str, str], set[str]] = {}
    for path in _credential_files(roots):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern in PAIR_PATTERNS:
            for match in pattern.finditer(text):
                user_id, api_key = match.groups()
                pairs.setdefault((user_id.lower(), api_key.lower()), set()).add(str(path))
    return pairs


def child_probe() -> None:
    from lightning_sdk import Teamspace
    from lightning_sdk.api.user_api import UserApi

    api = UserApi()
    client = api._client
    user = client.auth_service_get_user()
    organizations = api._get_organizations_for_authed_user()
    memberships = api._get_all_teamspace_memberships(user.id)
    organization_by_id = {item.id: item for item in organizations}
    teamspaces = []
    seen = set()
    for membership in memberships:
        if membership.project_id in seen or membership.name == "general":
            continue
        seen.add(membership.project_id)
        organization = organization_by_id.get(membership.owner_id)
        if organization is None:
            continue
        storage_bytes = None
        balance = None
        try:
            teamspace = Teamspace(membership.name, org=organization.name)
            storage_bytes = int(teamspace._teamspace.current_storage_bytes or 0)
            balance = client.billing_service_get_project_balance(teamspace.id).balance
        except Exception:
            pass
        teamspaces.append(
            {
                "organization": organization.name,
                "teamspace": membership.name,
                "teamspace_id": membership.project_id,
                "storage_bytes": storage_bytes,
                "balance": balance,
            }
        )
    print(
        json.dumps(
            {
                "user_id": user.id,
                "username": user.username,
                "teamspaces": teamspaces,
            }
        )
    )


def parent_probe(roots: tuple[Path, ...]) -> None:
    discovered = discover_pairs(roots)
    rows = []
    for (user_id, api_key), sources in discovered.items():
        env = os.environ.copy()
        env["LIGHTNING_USER_ID"] = user_id
        env["LIGHTNING_API_KEY"] = api_key
        completed = subprocess.run(
            [sys.executable, __file__, "--child"],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        fingerprint = hashlib.sha256(api_key.encode("ascii")).hexdigest()[:10]
        if completed.returncode:
            rows.append(
                {
                    "requested_user_id": user_id,
                    "credential_fingerprint": fingerprint,
                    "source_count": len(sources),
                    "valid": False,
                    "error": _sanitize_error(completed.stderr),
                }
            )
            continue
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        payload.update(
            {
                "credential_fingerprint": fingerprint,
                "source_count": len(sources),
                "valid": True,
            }
        )
        rows.append(payload)

    unique = {}
    for row in rows:
        key = row.get("user_id", row.get("requested_user_id"))
        existing = unique.get(key)
        if existing is None or (row.get("valid") and not existing.get("valid")):
            unique[key] = row
    print(json.dumps({"credential_pairs": len(discovered), "accounts": list(unique.values())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--roots", nargs="*")
    args = parser.parse_args()
    if args.child:
        child_probe()
    else:
        roots = tuple(Path(item) for item in args.roots) if args.roots else DEFAULT_ROOTS
        parent_probe(roots)
