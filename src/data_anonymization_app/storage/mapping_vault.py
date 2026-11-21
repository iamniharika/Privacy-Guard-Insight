from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VaultData = dict[str, Any]


def _empty_vault() -> VaultData:
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "vault_version": 1,
        "created_at": now_iso,
        "last_updated": now_iso,
        "token_format": "token_{counter:06d}",
        "columns": {},
    }


def load_mapping_vault(vault_path: str | Path) -> VaultData:
    path = Path(vault_path)

    if not path.exists():
        return _empty_vault()

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    columns = payload.get("columns")
    if not isinstance(columns, dict):
        return _empty_vault()

    normalized_columns: dict[str, dict[str, str]] = {}

    for column_name, mapping in columns.items():
        if not isinstance(mapping, dict):
            continue

        normalized_columns[str(column_name)] = {
            str(original_value): str(token)
            for original_value, token in mapping.items()
        }

    # Start with a fresh vault structure (with metadata)
    vault = _empty_vault()

    # Insert normalized column mappings
    vault["columns"] = normalized_columns

    # Preserve metadata if it exists
    if isinstance(payload.get("vault_version"), int):
        vault["vault_version"] = payload["vault_version"]

    if isinstance(payload.get("created_at"), str):
        vault["created_at"] = payload["created_at"]

    if isinstance(payload.get("last_updated"), str):
        vault["last_updated"] = payload["last_updated"]

    if isinstance(payload.get("token_format"), str):
        vault["token_format"] = payload["token_format"]

    return vault


def save_mapping_vault(vault_path: str | Path, vault_data: VaultData) -> None:
    path = Path(vault_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure metadata exists even for older vault files
    if "vault_version" not in vault_data:
        vault_data["vault_version"] = 1
    if "created_at" not in vault_data:
        vault_data["created_at"] = datetime.now(timezone.utc).isoformat()
    if "token_format" not in vault_data:
        vault_data["token_format"] = "token_{counter:06d}"

    # Update the last_updated timestamp
    vault_data["last_updated"] = datetime.now(timezone.utc).isoformat()

    with path.open("w", encoding="utf-8") as file:
        json.dump(vault_data, file, ensure_ascii=True, indent=2, sort_keys=True)


def _next_token_index(column_mapping: dict[str, str]) -> int:
    max_index = 0
    for token in column_mapping.values():
        if not token.startswith("token_"):
            continue
        suffix = token.removeprefix("token_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def get_or_create_token(
    vault_data: VaultData,
    column_name: str,
    original_value: str,
) -> str:
    columns = vault_data.setdefault("columns", {})
    column_mapping = columns.setdefault(str(column_name), {})
    if original_value in column_mapping:
        return column_mapping[original_value]

    next_index = _next_token_index(column_mapping)
    token = f"token_{next_index:06d}"
    column_mapping[original_value] = token
    return token

